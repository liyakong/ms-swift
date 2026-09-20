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
import json
import os
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

from .calculator import plan_segment
from .constants import CONFIG_ENV, SEGMENT_CONFIG_FILENAME, STATE_FILENAME, STOP_FILENAME, count_visible_devices
from .plan_builder import dataset_fingerprint, iter_plan_records, load_raw_samples
from .slicer import write_slice
from .state import (atomic_write_json, find_latest_committed_state, load_segment_config, load_state_file)

logger = get_logger()

PREV_CHECKPOINT_ENV = 'SWIFT_ELASTIC_RELAY_PREV_CHECKPOINT'
ROUTE = 'sft'
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


def _dispatch(argv: List[str]) -> int:
    from swift.cli.main import cli_main
    sys.argv = [sys.argv[0], ROUTE, *argv]
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


def _prepare(argv: List[str]) -> int:
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
    if prev_state is not None:
        _verify_recipe(prev_state, {
            'seed': seed,
            'num_train_epochs': epochs,
            'dataset_size': raw_size,
            'raw_dataset_fingerprint': raw_fp,
        })
    cursor = int(prev_state['committed_cursor']) if prev_state else int(config.get('committed_cursor', 0))

    num_gpus, pdb, accum = _resolve_parallelism(argv, config)
    global_batch = num_gpus * pdb * accum
    segment = plan_segment(cursor, total_units, global_batch, checkpoint_path=prev_path)
    if segment is None:
        logger.info('elastic_relay: the training plan is already fully consumed; nothing to train.')
        return 0

    chain = list(config.get('segment_chain') or (prev_state or {}).get('segment_chain') or [])
    if prev_state and prev_state.get('segment_id') and prev_state['segment_id'] not in chain:
        chain.append(prev_state['segment_id'])
    segment_id = config.get('segment_id') or f'seg{len(chain) + 1:02d}'

    slice_dir = os.path.join(output_dir, 'elastic_relay')
    slice_path = os.path.join(slice_dir, f'{segment_id}.jsonl')
    write_slice(
        iter_plan_records(samples, epochs, seed, cursor, total_units),
        cursor,
        total_units,
        segment['pad_count'],
        slice_path,
        template_record=samples[0],
        source=raw_path)

    stop_file = config.get('stop_file') or os.path.join(output_dir, STOP_FILENAME)
    plan_fingerprint = config.get('plan_fingerprint') or f'{raw_fp}:seed={seed}:epochs={epochs}'
    resolved = dict(config)
    resolved.update({
        'raw_dataset_path': raw_path,
        'raw_dataset_fingerprint': raw_fp,
        'dataset_size': raw_size,
        'seed': seed,
        'num_train_epochs': epochs,
        'total_units': total_units,
        'plan_fingerprint': plan_fingerprint,
        'segment_id': segment_id,
        'segment_chain': chain,
        'segment_start_cursor': cursor,
        'num_gpus': num_gpus,
        'per_device_batch': pdb,
        'grad_accum': accum,
        'stop_file': stop_file,
        'output_dir': output_dir,
    })
    os.environ[CONFIG_ENV] = json.dumps(resolved, ensure_ascii=False)
    atomic_write_json(os.path.join(output_dir, SEGMENT_CONFIG_FILENAME), resolved)

    # Essential correctness args, set regardless of how the platform built the command.
    if prev_path:
        _set_flag(argv, '--resume_from_checkpoint', prev_path)
        _set_flag(argv, '--resume_only_model', 'true')
        _set_flag(argv, '--ignore_data_skip', 'true')
    _set_flag(argv, '--output_dir', output_dir)
    _set_flag(argv, '--num_train_epochs', '1')
    for name, value in (('--dataset_shuffle', 'false'), ('--train_dataloader_shuffle', 'false'),
                        ('--group_by_length', 'false'), ('--load_args', 'false'),
                        ('--dataloader_drop_last', 'true')):
        _set_flag(argv, name, value)
    if segment['pad_count']:
        if not _get_flag(argv, '--loss_type'):
            _set_flag(argv, '--loss_type', 'elastic_cross_entropy')
    _merge_list_flag(argv, '--external_plugins', [_plugin_path()])
    _merge_list_flag(argv, '--callbacks', ['elastic_stop', 'elastic_state'])
    argv.extend(['--dataset', slice_path])

    _clear_stale_stop_file(stop_file)
    _install_stop_handler(stop_file)
    logger.info(f'elastic_relay[{segment_id}]: cursor={cursor}/{total_units}, pad={segment["pad_count"]}, '
                f'gb={global_batch}, slice={slice_path}')
    return _dispatch(argv)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == ROUTE:
        argv = argv[1:]
    if os.environ.get('SWIFT_ELASTIC_RELAY_PASSTHROUGH'):
        logger.warning('elastic_relay: pass-through mode; forwarding to swift unchanged.')
        return _dispatch(argv)
    return _prepare(argv)


if __name__ == '__main__':
    sys.exit(main())
