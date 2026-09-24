# Copyright (c) ModelScope Contributors. All rights reserved.
"""Self-contained entry point for one elastic-relay segment job.

The platform keeps passing the *original complete dataset* and each segment runs in a
fresh container with no shared root, so the plan cannot be persisted. This entry instead
reconstructs ``plan[cursor:total]`` deterministically from ``(raw, seed, epochs)`` inside
the job, rewrites ``--dataset`` to the slice, and then delegates to ``swift sft``.

It also acts as PID 1: Kubernetes sends ``SIGTERM`` to PID 1 on pod deletion; the handler
touches the stop file, which the in-training ``elastic_stop`` callback turns into a
checkpoint save at the next optimizer-step boundary (regardless of ``save_steps``).

Invocation::

    python -m swift.elastic_relay.runner --model ... --dataset /data/raw.jsonl ...

All configuration comes from the ``SWIFT_ELASTIC_RELAY_CONFIG`` environment variable
(inline JSON or a file path).
"""
import functools
import glob
import json
import os
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

from .calculator import plan_segment
from .constants import CONFIG_ENV, SEGMENT_CONFIG_FILENAME, STATE_FILENAME, STOP_FILENAME, count_visible_devices
from .grpo import resolve_generation_params, round_aligned_end
from .plan_builder import dataset_fingerprint, iter_plan_records, load_raw_samples
from .slicer import make_dummy_sample, write_slice
from .state import (atomic_write_json, find_latest_committed_state, load_segment_config, load_state_file,
                    scrub_external_plugins)

logger = get_logger()

PREV_CHECKPOINT_ENV = 'SWIFT_ELASTIC_RELAY_PREV_CHECKPOINT'
DEFAULT_ROUTE = 'sft'
ROUTES = {'sft', 'rlhf', 'pt'}
# rlhf_type values that use the paired DPO-style pipeline (swift normalizes simpo -> cpo).
DPO_TYPES = {'dpo', 'orpo', 'cpo'}
_LIST_FLAGS = {'--dataset', '--callbacks', '--external_plugins'}


# --------------------------------------------------------------------------------------
# argv helpers
# --------------------------------------------------------------------------------------
def _pop_flag(argv: List[str], name: str) -> List[str]:
    values: List[str] = []
    out: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == name:
            i += 1
            while i < len(argv) and not argv[i].startswith('--'):
                values.append(argv[i])
                i += 1
            continue
        if token.startswith(name + '='):
            values.append(token.split('=', 1)[1])
            i += 1
            continue
        out.append(token)
        i += 1
    argv[:] = out
    return values


def _get_flag(argv: List[str], name: str, default: Optional[str] = None) -> Optional[str]:
    values: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == name:
            i += 1
            while i < len(argv) and not argv[i].startswith('--'):
                values.append(argv[i])
                i += 1
            continue
        if argv[i].startswith(name + '='):
            values.append(argv[i].split('=', 1)[1])
        i += 1
    return values[-1] if values else default


def _get_list_flag(argv: List[str], name: str) -> List[str]:
    """Collect every value of a (possibly repeated / multi-value) flag without mutating argv."""
    values: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == name:
            i += 1
            while i < len(argv) and not argv[i].startswith('--'):
                values.append(argv[i])
                i += 1
            continue
        if argv[i].startswith(name + '='):
            values.append(argv[i].split('=', 1)[1])
        i += 1
    return values


def _set_flag(argv: List[str], name: str, value: str) -> None:
    _pop_flag(argv, name)
    argv.extend([name, value])


def _merge_list_flag(argv: List[str], name: str, additions: List[str]) -> None:
    values = _pop_flag(argv, name)
    merged = list(dict.fromkeys(values + additions))
    argv.extend([name] + merged)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _plugin_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plugin.py')


def _resolve_prev_state(prev: Optional[str], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read the committed cursor from the platform's resume-checkpoint path.

    Accepts a ``platform_state.json`` file, a ``checkpoint-N`` directory, or a run output
    directory containing ``checkpoint-*`` dirs.
    """
    if not prev:
        return None
    state = None
    if os.path.isfile(prev):
        state = load_state_file(prev)
    elif os.path.isdir(prev):
        if os.path.exists(os.path.join(prev, STATE_FILENAME)):
            state = load_state_file(prev)
        else:
            state = find_latest_committed_state([prev])
            if state is not None:
                return state
    if state is not None:
        state.setdefault('checkpoint_path', prev)
        return state

    if 'committed_cursor' in config:
        logger.warning(f'elastic_relay: no platform_state.json found under {prev}; using the explicit '
                       f'committed_cursor={config["committed_cursor"]} from the config.')
        fallback = dict(config)
        fallback.setdefault('checkpoint_path', prev)
        return fallback
    raise RuntimeError(
        f'elastic_relay: cannot determine the training-plan cursor because {prev} has no '
        'platform_state.json. This checkpoint was not produced by elastic_relay. Point the resume '
        'path at an elastic_relay checkpoint, or set `committed_cursor` in the config to bootstrap.')


def _clear_stale_stop_file(stop_file: str) -> None:
    """Remove a STOP file left over from the previous segment.

    The stop file lives at a stable path under the (shared) output dir, so a file created
    to stop segment N would otherwise stop segment N+1 on its very first step.
    """
    if stop_file and os.path.exists(stop_file):
        try:
            os.remove(stop_file)
            logger.warning(f'elastic_relay: removed stale stop file {stop_file} left by a previous segment.')
        except OSError as e:  # pragma: no cover - best effort
            logger.error(f'elastic_relay: failed to remove stale stop file {stop_file}: {e}')


def _response_keys(argv: List[str], config: Dict[str, Any]) -> List[str]:
    """User-mapped dataset columns that swift treats as the response (from ``--columns``).

    Every built-in auto-mapped name is already handled by ``slicer.make_dummy_sample``; this
    adds the user's own mappings so a custom column name is neutralized in padding rows too.
    """
    columns = _get_flag(argv, '--columns') or config.get('columns')
    if not columns:
        return []
    from swift.utils import json_parse_to_dict
    try:
        mapping = json_parse_to_dict(columns)
    except Exception:  # noqa: BLE001 - a malformed mapping is swift's problem, not ours
        return []
    if not isinstance(mapping, dict):
        return []
    return [str(src) for src, dst in mapping.items() if str(dst) == 'response']


def _strip_external_plugins(output_dir: str) -> None:
    """Drop the elastic-relay plugin path from every saved ``args.json`` (portable checkpoints).

    See :func:`swift.elastic_relay.state.scrub_external_plugins`. Runs once per segment after the
    training subprocess exits, covering the top-level and every ``checkpoint-*/args.json``.
    """
    paths = glob.glob(os.path.join(output_dir, '**', 'args.json'), recursive=True)
    scrub_external_plugins(paths)


def _install_stop_handler(stop_file: str) -> None:
    def handler(signum, frame):  # noqa: ARG001
        try:
            os.makedirs(os.path.dirname(os.path.abspath(stop_file)), exist_ok=True)
            with open(stop_file, 'w', encoding='utf-8') as f:
                f.write(str(signum))
            logger.warning(f'elastic_relay: received signal {signum}; wrote stop file {stop_file}. '
                           'The trainer will checkpoint at the next step boundary.')
        except Exception as e:  # pragma: no cover - best effort
            logger.error(f'elastic_relay: failed to write stop file {stop_file}: {e}')

    if not stop_file:
        return
    for name in ('SIGTERM', 'SIGINT', 'SIGUSR1', 'SIGUSR2'):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, handler)
        except Exception:  # pragma: no cover - platform dependent
            pass


def _dispatch(argv: List[str], route: str = DEFAULT_ROUTE) -> int:
    from swift.cli.main import cli_main
    sys.argv = [sys.argv[0], route, *argv]
    cli_main()
    return 0


def _optional_config() -> Dict[str, Any]:
    """Config is optional: everything can be derived from argv + the previous checkpoint."""
    value = os.environ.get(CONFIG_ENV)
    if not value:
        return {}
    return load_segment_config(value)


def _deepspeed_grad_accum(argv: List[str], config: Dict[str, Any]) -> Optional[int]:
    """Return the numeric gradient_accumulation_steps from a DeepSpeed config, if it wins.

    DeepSpeed's value overrides accelerate's plugin value when they differ, so the global
    batch must be computed from it. Swift's presets use ``"auto"`` (meaning "use the plugin
    value"), in which case we keep the caller-derived value.
    """
    ds = _get_flag(argv, '--deepspeed')
    data = None
    if ds is None:
        ds_config = config.get('deepspeed')
        if isinstance(ds_config, dict):
            data = ds_config
        else:
            ds = ds_config
    if data is None:
        if not ds:
            return None
        ds = str(ds).strip()
        if ds in {'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'}:
            return None
        try:
            if os.path.exists(ds):
                with open(ds, encoding='utf-8') as f:
                    data = json.load(f)
            elif ds.startswith('{'):
                data = json.loads(ds)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f'elastic_relay: failed to parse --deepspeed config {ds!r}: {e}')
            return None
    if isinstance(data, dict):
        value = data.get('gradient_accumulation_steps')
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _resolve_parallelism(argv: List[str], config: Dict[str, Any]) -> Any:
    pdb = int(_get_flag(argv, '--per_device_train_batch_size') or config.get('per_device_batch') or 1)
    num_gpus = os.environ.get('NPROC_PER_NODE') or config.get('num_gpus')
    if not num_gpus:
        # Works for Ascend (ASCEND_RT_VISIBLE_DEVICES), CUDA, and other NPU/GPU runtimes.
        num_gpus = count_visible_devices() or 1
    num_gpus = int(num_gpus)

    accum = _get_flag(argv, '--gradient_accumulation_steps') or config.get('grad_accum')
    if accum:
        accum = int(accum)
    else:
        # Mirror TrainArgumentsMixin.__post_init__: aim for a global batch of ~16.
        accum = max(1, -(-16 // (pdb * max(num_gpus, 1))))

    ds_accum = _deepspeed_grad_accum(argv, config)
    if ds_accum is not None and ds_accum != accum:
        logger.warning(f'elastic_relay: DeepSpeed config sets gradient_accumulation_steps={ds_accum}; '
                       f'using it instead of {accum} for the global-batch computation.')
        accum = ds_accum
    return num_gpus, pdb, accum


def _preflight() -> None:
    """Ensure the training subprocess will import a `swift` that contains `elastic_relay`.

    The parent process may resolve `swift` to this repo while the child (torchrun executing
    ``swift/cli/sft.py``) resolves it to an installed copy in site-packages that lacks the
    new subpackage. Probe from the same directory the child uses to reproduce that.
    """
    swift_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../swift
    repo_root = os.path.dirname(swift_dir)
    probe_cwd = os.path.join(swift_dir, 'cli')
    result = subprocess.run(
        [sys.executable, '-c', 'import swift.elastic_relay'],
        cwd=probe_cwd,
        env=os.environ,
        capture_output=True,
        text=True)
    if result.returncode != 0:
        raise RuntimeError(
            'elastic_relay: the `swift` package imported by the training subprocess does not contain '
            f'`elastic_relay` (probe from {probe_cwd} failed).\n{result.stderr.strip()}\n'
            f'Fix: run `pip install -e {repo_root}` once, or `export PYTHONPATH={repo_root}` before launching.')


def _check_unsupported(dataset_args: List[str], argv: List[str], config: Dict[str, Any]) -> None:
    """Fail loudly instead of silently dropping data or changing the unit definition."""
    if len(dataset_args) > 1:
        raise NotImplementedError(
            'elastic_relay supports a single --dataset (one file, or one directory containing all data files). '
            f'Got {len(dataset_args)}: {dataset_args}. Please merge them into one directory upstream, or use a '
            'pre-materialized dataset.')
    if dataset_args and '#' in dataset_args[0]:
        raise NotImplementedError('elastic_relay does not support the swift `dataset#sample` syntax; the plan must '
                                  'cover the full dataset deterministically. Remove the `#...` suffix.')

    truthy = {'true', 'True', '1', 'yes'}
    streaming = _get_flag(argv, '--streaming') or config.get('streaming')
    packing = _get_flag(argv, '--packing') or config.get('packing')
    if str(streaming) in truthy:
        raise NotImplementedError('elastic_relay does not support streaming datasets (no fixed length).')
    if str(packing) in truthy:
        raise NotImplementedError('elastic_relay requires packing to be disabled (it reorders samples).')
    if _get_flag(argv, '--interleave_prob') or config.get('interleave_prob'):
        raise NotImplementedError('elastic_relay does not support --interleave_prob; the plan order must be fixed.')


def _verify_recipe(prev_state: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Fail loudly if plan-defining parameters differ from the previous segment."""
    diffs = []
    for key, value in current.items():
        previous = prev_state.get(key)
        if previous is not None and str(previous) != str(value):
            diffs.append(f'{key}: {previous} -> {value}')
    if diffs:
        raise RuntimeError('elastic_relay: the training recipe changed between segments, which would invalidate '
                           f'the cursor. {diffs}')


def _check_grpo_constraints(argv: List[str], config: Dict[str, Any]) -> None:
    """Reject GRPO configs that would pull prompts from outside the frozen plan."""
    truncation = _get_flag(argv, '--truncation_strategy') or config.get('truncation_strategy')
    if truncation != 'left':
        raise NotImplementedError(
            "elastic_relay[grpo]: `--truncation_strategy left` is required. The 'delete'/'raise' path "
            'resamples truncated prompts from a separate iterator (outside the plan), which breaks the '
            'no-repeat/no-omit guarantee.')
    dynamic = _get_flag(argv, '--dynamic_sample') or config.get('dynamic_sample')
    if str(dynamic) in {'true', 'True', '1'}:
        raise NotImplementedError('elastic_relay[grpo]: `--dynamic_sample false` is required (it resamples '
                                  'zero-variance groups from outside the plan).')


def _check_dpo_constraints(argv: List[str], config: Dict[str, Any]) -> None:
    """Reject DPO configs whose batch layout breaks the padding-row mask."""
    padding_free = _get_flag(argv, '--padding_free') or config.get('padding_free')
    if str(padding_free) in {'true', 'True', '1', 'yes'}:
        raise NotImplementedError(
            'elastic_relay[dpo]: `--padding_free false` is required. The padding-row mask is derived from '
            'the 2D `loss_mask`; padding-free flattens the batch into cu_seqlens.')


def _dpo_recipe(argv: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """DPO hyper-parameters that must stay fixed across segments (they change the objective)."""

    def flag(name: str, key: str) -> Any:
        return _get_flag(argv, name) or config.get(key)

    return {
        'dpo_rlhf_type': flag('--rlhf_type', 'rlhf_type'),
        'beta': flag('--beta', 'beta'),
        'loss_type': flag('--loss_type', 'loss_type'),
        'loss_weights': flag('--loss_weights', 'loss_weights'),
        'rpo_alpha': flag('--rpo_alpha', 'rpo_alpha'),
        'ld_alpha': flag('--ld_alpha', 'ld_alpha'),
        'label_smoothing': flag('--label_smoothing', 'label_smoothing'),
        'reference_free': flag('--reference_free', 'reference_free'),
        'ref_model': flag('--ref_model', 'ref_model'),
        'ref_adapter_name': flag('--ref_adapter_name', 'ref_adapter_name'),
        'sync_ref_model': flag('--sync_ref_model', 'sync_ref_model'),
        'precompute_ref_log_probs': flag('--precompute_ref_log_probs', 'precompute_ref_log_probs'),
    }


def _grpo_rollout_recipe(argv: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Rollout backend + sampling parameters: must stay consistent across segments.

    Changing these does not break the data cursor, but it changes the generation
    distribution and therefore the training semantics, so we lock them.
    """
    use_vllm = str(_get_flag(argv, '--use_vllm') or config.get('use_vllm') or 'false')
    vllm_enabled = use_vllm in ('true', 'True', '1')
    vllm_mode = _get_flag(argv, '--vllm_mode') or config.get('vllm_mode') or 'colocate'
    tp = int(_get_flag(argv, '--vllm_tensor_parallel_size') or config.get('vllm_tensor_parallel_size') or 1)
    world = int(os.environ.get('NPROC_PER_NODE') or config.get('num_gpus') or count_visible_devices() or 1)
    if vllm_enabled and vllm_mode == 'colocate' and tp > 0 and world % tp != 0:
        raise ValueError(f'elastic_relay[grpo]: --vllm_tensor_parallel_size ({tp}) must divide world_size '
                         f'({world}) in colocate mode.')
    return {
        'use_vllm': use_vllm,
        'vllm_mode': vllm_mode if vllm_enabled else None,
        'temperature': str(_get_flag(argv, '--temperature') or config.get('temperature') or ''),
        'top_p': str(_get_flag(argv, '--top_p') or config.get('top_p') or ''),
        'top_k': str(_get_flag(argv, '--top_k') or config.get('top_k') or ''),
        'max_completion_length': str(
            _get_flag(argv, '--max_completion_length') or config.get('max_completion_length') or ''),
    }


def _warn_grpo_deepspeed_offload(argv: List[str], config: Dict[str, Any]) -> None:
    """Warn about the known DeepSpeed ZeRO-3 + parameter-offload bug with GRPO's extra forwards."""
    truthy = ('true', 'True', '1')
    ds = str(_get_flag(argv, '--deepspeed') or config.get('deepspeed') or '')
    offload = (str(_get_flag(argv, '--offload_model') or config.get('offload_model') or 'false') in truthy
               or str(_get_flag(argv, '--offload_optimizer') or config.get('offload_optimizer') or 'false') in truthy
               or 'offload' in ds)
    if offload and ds:
        logger.warning('elastic_relay[grpo]: DeepSpeed parameter/optimizer offload is enabled. Some DeepSpeed '
                       'versions hit a ZeRO-3 parameter_offload "pop from an empty deque" bug with GRPO\'s extra '
                       'local forwards. It is version-dependent: if it happens, upgrade `deepspeed`, or use '
                       '`--deepspeed zero2` / zero3 without offload.')


def _resolve_grpo_generation(argv: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    pdb = int(_get_flag(argv, '--per_device_train_batch_size') or config.get('per_device_batch') or 1)
    world = os.environ.get('NPROC_PER_NODE') or config.get('num_gpus')
    if not world:
        world = count_visible_devices() or 1
    grad_accum = _get_flag(argv, '--gradient_accumulation_steps') or config.get('grad_accum')
    num_generations = int(_get_flag(argv, '--num_generations') or config.get('num_generations') or 8)
    num_iterations = int(_get_flag(argv, '--num_iterations') or config.get('num_iterations') or 1)
    steps_per_generation = _get_flag(argv, '--steps_per_generation') or config.get('steps_per_generation')
    generation_batch_size = _get_flag(argv, '--generation_batch_size') or config.get('generation_batch_size')

    # GRPO's generation-batch accounting uses the *args* gradient_accumulation_steps (default 1
    # for GRPO). A DeepSpeed config with a hardcoded numeric value would silently override it and
    # break the cursor math, so require consistency (presets use "auto" and return None here).
    ds_accum = _deepspeed_grad_accum(argv, config)
    effective_accum = int(grad_accum) if grad_accum else 1
    if ds_accum is not None and ds_accum != effective_accum:
        raise NotImplementedError(
            f'elastic_relay[grpo]: DeepSpeed config sets gradient_accumulation_steps={ds_accum}, but the training '
            f'args use {effective_accum}; GRPO generation-batch accounting assumes they match. Use a DeepSpeed '
            'preset (`--deepspeed zero2`/`zero3`, whose config uses "auto"), or set '
            '`--gradient_accumulation_steps` equal to the DeepSpeed value.')

    return resolve_generation_params(
        pdb,
        int(world),
        num_generations,
        num_iterations,
        int(grad_accum) if grad_accum else None,
        int(steps_per_generation) if steps_per_generation else None,
        int(generation_batch_size) if generation_batch_size else None)


def _grpo_recipe(seed: int, epochs: int, raw_size: int, raw_fp: str, generation: Dict[str, Any]) -> Dict[str, Any]:
    """Plan/generation-defining fields that must not change across segments.

    ``world_size`` and everything derived from it (``steps_per_generation``,
    ``generation_batch_size``, ``prompts_per_round``) are intentionally excluded: changing
    the card count is allowed, and the prompt order is plan order regardless of P.
    """
    return {
        'mode': 'grpo',
        'seed': seed,
        'num_train_epochs': epochs,
        'dataset_size': raw_size,
        'raw_dataset_fingerprint': raw_fp,
        'num_generations': generation['num_generations'],
        'num_iterations': generation['num_iterations'],
        'per_device_batch': generation['per_device_batch'],
        'gradient_accumulation_steps': generation['gradient_accumulation_steps'],
    }


def _grpo_ref_recipe(argv: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Reference-model identity, which must be pinned across segments.

    ``ref_model`` is excluded from the model-resume path, but GRPO's KL uses it. For
    full-parameter GRPO the framework defaults ``ref_model = self.model`` and, on resume
    with ``resume_only_model``, ``self.model`` is the checkpoint -> the reference would
    drift each segment. So an explicit ``--ref_model`` is required there.
    """
    tuner_type = _get_flag(argv, '--tuner_type') or config.get('tuner_type') or 'lora'
    beta_raw = _get_flag(argv, '--beta') or config.get('beta')
    beta = float(beta_raw) if beta_raw not in (None, '') else 0.04
    ref_model = _get_flag(argv, '--ref_model') or config.get('ref_model')
    ref_adapters = sorted(_get_list_flag(argv, '--ref_adapters') or config.get('ref_adapters') or [])
    sync_ref_model = str(_get_flag(argv, '--sync_ref_model') or config.get('sync_ref_model') or 'false')
    if beta != 0.0 and tuner_type == 'full' and not ref_model:
        raise NotImplementedError(
            'elastic_relay[grpo]: full-parameter GRPO with beta != 0 requires an explicit `--ref_model`. '
            'Otherwise the framework sets ref_model = the resumed checkpoint, so the KL reference drifts at '
            'every segment. Pass the fixed base/initial model via `--ref_model`.')
    return {
        'tuner_type': tuner_type,
        'beta': beta,
        'ref_model': ref_model,
        'ref_adapters': ref_adapters,
        'sync_ref_model': sync_ref_model,
    }


def _prepare(argv: List[str], route: str = DEFAULT_ROUTE) -> int:
    _preflight()
    config = _optional_config()
    output_dir = _get_flag(argv, '--output_dir') or config.get('output_dir')
    if not output_dir:
        raise ValueError('elastic_relay: `--output_dir` is required. Refusing to default to the current '
                         'directory, which would scatter slice/checkpoint output into the working tree.')
    output_dir = os.path.abspath(output_dir)

    dataset_args = _pop_flag(argv, '--dataset')
    _check_unsupported(dataset_args, argv, config)
    raw_path = (dataset_args[0] if dataset_args else None) or config.get('raw_dataset_path')
    if not raw_path:
        raise ValueError('elastic_relay: no raw dataset found (--dataset or config.raw_dataset_path).')
    raw_path = os.path.abspath(raw_path)

    samples = load_raw_samples(raw_path)
    raw_size = len(samples)
    raw_fp = dataset_fingerprint(raw_path)
    epochs = int(_get_flag(argv, '--num_train_epochs') or config.get('num_train_epochs') or 1)
    seed = int(_get_flag(argv, '--seed') or config.get('seed') or 42)
    configured_size = config.get('dataset_size')
    if configured_size is not None and int(configured_size) != raw_size:
        raise RuntimeError(f'elastic_relay: raw dataset size changed: config={configured_size}, actual={raw_size} '
                           f'({raw_path}).')
    total_units = int(config.get('total_units') or raw_size * epochs)

    prev_path = _pop_flag(argv, '--resume_from_checkpoint')
    prev_path = (prev_path[-1] if prev_path else None) or config.get('prev_checkpoint') \
        or os.environ.get(PREV_CHECKPOINT_ENV)
    prev_state = _resolve_prev_state(prev_path, config)
    cursor = int(prev_state['committed_cursor']) if prev_state else int(config.get('committed_cursor', 0))

    rlhf_type = _get_flag(argv, '--rlhf_type') or config.get('rlhf_type')
    is_grpo = route == 'rlhf' and rlhf_type == 'grpo'
    is_dpo = route == 'rlhf' and rlhf_type in DPO_TYPES
    generation: Optional[Dict[str, Any]] = None
    if is_grpo:
        _check_grpo_constraints(argv, config)
        generation = _resolve_grpo_generation(argv, config)
        ref_recipe = _grpo_ref_recipe(argv, config)
        rollout_recipe = _grpo_rollout_recipe(argv, config)
        _warn_grpo_deepspeed_offload(argv, config)
        recipe = _grpo_recipe(seed, epochs, raw_size, raw_fp, generation)
        recipe.update(ref_recipe)
        recipe.update(rollout_recipe)
        if prev_state is not None:
            _verify_recipe(prev_state, recipe)
        end_cursor = round_aligned_end(cursor, total_units, generation['prompts_per_round'])
        segment = {
            'start_cursor': cursor,
            'end_cursor': end_cursor,
            'remaining_units': end_cursor - cursor,
            'pad_count': 0,  # GRPO padding would generate real rollouts; slice whole rounds instead.
            'estimated_steps': (end_cursor - cursor) // generation['prompts_per_round'],
            'checkpoint_path': prev_path,
            'new_lr': None,
            'warmup_steps': 0,
        }
        global_batch = generation['generation_batch_size']
        mode = 'grpo'
        if segment['remaining_units'] <= 0:
            logger.info('elastic_relay[grpo]: fewer than one generation round remains; DONE.')
            return 0
    else:
        recipe = {
            'mode': 'sft',
            'seed': seed,
            'num_train_epochs': epochs,
            'dataset_size': raw_size,
            'raw_dataset_fingerprint': raw_fp,
        }
        if is_dpo:
            _check_dpo_constraints(argv, config)
            recipe.update(_dpo_recipe(argv, config))
        if prev_state is not None:
            _verify_recipe(prev_state, recipe)
        num_gpus, pdb, accum = _resolve_parallelism(argv, config)
        global_batch = num_gpus * pdb * accum
        segment = plan_segment(cursor, total_units, global_batch, checkpoint_path=prev_path)
        if segment is None:
            logger.info('elastic_relay: the training plan is already fully consumed; nothing to train.')
            return 0
        mode = 'sft'

    chain = list(config.get('segment_chain') or (prev_state or {}).get('segment_chain') or [])
    if prev_state and prev_state.get('segment_id') and prev_state['segment_id'] not in chain:
        chain.append(prev_state['segment_id'])
    segment_id = config.get('segment_id') or f'seg{len(chain) + 1:02d}'

    slice_dir = os.path.join(output_dir, 'elastic_relay')
    slice_path = os.path.join(slice_dir, f'{segment_id}.jsonl')
    dummy_factory = functools.partial(
        make_dummy_sample,
        response_keys=_response_keys(argv, config),
        # PT uses no chat template + `loss_scale='all'`, so the pad row must carry `loss: false`.
        no_chat_template=(route == 'pt'))
    write_slice(
        iter_plan_records(samples, epochs, seed, cursor, segment['end_cursor']),
        cursor,
        segment['end_cursor'],
        segment['pad_count'],
        slice_path,
        template_record=samples[0] if segment['pad_count'] else None,
        dummy_factory=dummy_factory,
        source=raw_path)

    stop_file = config.get('stop_file') or os.path.join(output_dir, STOP_FILENAME)
    plan_fingerprint = config.get('plan_fingerprint') or f'{raw_fp}:seed={seed}:epochs={epochs}'
    resolved = dict(config)
    resolved.update({
        'mode': mode,
        'raw_dataset_path': raw_path,
        'raw_dataset_fingerprint': raw_fp,
        'dataset_size': raw_size,
        'seed': seed,
        'num_train_epochs': epochs,
        'total_units': total_units,
        'segment_end_cursor': segment['end_cursor'],
        'plan_fingerprint': plan_fingerprint,
        'segment_id': segment_id,
        'segment_chain': chain,
        'segment_start_cursor': cursor,
        'stop_file': stop_file,
        'output_dir': output_dir,
    })
    if mode == 'grpo':
        resolved.update({
            'rlhf_type': 'grpo',
            'num_gpus': generation['world_size'],
            'per_device_batch': generation['per_device_batch'],
            'grad_accum': generation['gradient_accumulation_steps'],
            'num_generations': generation['num_generations'],
            'num_iterations': generation['num_iterations'],
            'steps_per_generation': generation['steps_per_generation'],
            'generation_batch_size': generation['generation_batch_size'],
            'global_batch_size': generation['global_batch_size'],
            'prompts_per_round': generation['prompts_per_round'],
            'optimizer_steps_per_round': generation['optimizer_steps_per_round'],
        })
        resolved.update(ref_recipe)
        resolved.update(rollout_recipe)
    else:
        resolved.update({
            'num_gpus': num_gpus,
            'per_device_batch': pdb,
            'grad_accum': accum,
        })
        if is_dpo:
            resolved.update({'rlhf_type': rlhf_type})
    os.environ[CONFIG_ENV] = json.dumps(resolved, ensure_ascii=False)
    atomic_write_json(os.path.join(output_dir, SEGMENT_CONFIG_FILENAME), resolved)

    # Essential correctness args, set regardless of how the platform built the command.
    if prev_path:
        _set_flag(argv, '--resume_from_checkpoint', prev_path)
        _set_flag(argv, '--resume_only_model', 'true')
        _set_flag(argv, '--ignore_data_skip', 'true')
    _set_flag(argv, '--output_dir', output_dir)
    _set_flag(argv, '--num_train_epochs', '1')
    if mode == 'grpo' and generation['optimizer_steps_per_round'] > 0:
        # Align periodic saves to generation-round boundaries so every save can be committed.
        osr = generation['optimizer_steps_per_round']
        requested = int(_get_flag(argv, '--save_steps') or config.get('save_steps') or 100)
        aligned = max(osr, (requested // osr) * osr)
        _set_flag(argv, '--save_steps', str(aligned))
    for name, value in (('--dataset_shuffle', 'false'), ('--train_dataloader_shuffle', 'false'),
                        ('--group_by_length', 'false'), ('--load_args', 'false'),
                        ('--dataloader_drop_last', 'true')):
        _set_flag(argv, name, value)
    if segment['pad_count'] and not is_dpo:
        # SFT pads with all `-100` rows (zero cross-entropy) and needs the DDP-safe loss.
        # DPO's `--loss_type` selects its own objective; padding is masked inside DPOTrainer.
        if not _get_flag(argv, '--loss_type'):
            _set_flag(argv, '--loss_type', 'elastic_cross_entropy')
    _merge_list_flag(argv, '--external_plugins', [_plugin_path()])
    _merge_list_flag(argv, '--callbacks', ['elastic_stop', 'elastic_state'])
    argv.extend(['--dataset', slice_path])

    _clear_stale_stop_file(stop_file)
    _install_stop_handler(stop_file)
    logger.info(f'elastic_relay[{segment_id}]: mode={mode}, cursor={cursor}/{total_units}, '
                f'end={segment["end_cursor"]}, pad={segment["pad_count"]}, gb={global_batch}, slice={slice_path}')
    code = _dispatch(argv, route)
    _strip_external_plugins(output_dir)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    route = DEFAULT_ROUTE
    if argv and argv[0] in ROUTES:
        route = argv[0]
        argv = argv[1:]
    if os.environ.get('SWIFT_ELASTIC_RELAY_PASSTHROUGH'):
        logger.warning('elastic_relay: pass-through mode; forwarding to swift unchanged.')
        return _dispatch(argv, route)
    return _prepare(argv, route)


if __name__ == '__main__':
    sys.exit(main())
