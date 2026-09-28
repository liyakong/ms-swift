# Copyright (c) ModelScope Contributors. All rights reserved.
"""Persistence helpers for the two pieces of elastic-relay state.

There are exactly two progress states:

* ``checkpoint``: how far the *model weights* have been trained.
* ``committed_cursor``: how far the *training plan* has been consumed.

They are intentionally decoupled: the cursor is the contract between segments and is
never inferred from the framework's ``global_step``.
"""
import glob
import json
import os
import time
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

from .constants import CONFIG_ENV, SEGMENT_CONFIG_FILENAME, STATE_FILENAME, STOP_FILENAME

logger = get_logger()


# --------------------------------------------------------------------------------------
# generic json io
# --------------------------------------------------------------------------------------
def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def scrub_external_plugins(paths: List[str]) -> List[str]:
    """Blank the environment-specific plugin paths inside saved ``args.json`` files.

    The runner injects ``--external_plugins <repo>/swift/elastic_relay/plugin.py``; swift
    persists *every* argument and copies ``args.json`` into each checkpoint, which makes the
    checkpoint non-portable: loading it in an image without that path dies in
    ``import_external_file`` (``assert os.path.isdir``). The plugin is only needed at training
    time (the runner re-injects it), so we blank these keys. Returns the cleaned paths.
    """
    keys = ('external_plugins', 'custom_register_path')
    stripped = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        changed = False
        for key in keys:
            if data.get(key):
                data[key] = []
                changed = True
        if changed:
            atomic_write_json(path, data)
            stripped.append(path)
    if stripped:
        logger.info(f'elastic_relay: blanked external_plugins in {len(stripped)} args.json: {stripped}')
    return stripped


# --------------------------------------------------------------------------------------
# per-segment config (consumed by callbacks through the `CONFIG_ENV` env var)
# --------------------------------------------------------------------------------------
def build_segment_config(plan_meta: Dict[str, Any],
                         segment_id: str,
                         segment_start_cursor: int,
                         num_gpus: int,
                         per_device_batch: int,
                         grad_accum: int,
                         segment_chain: Optional[List[str]] = None,
                         stop_file: Optional[str] = None,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
    return {
        'plan_fingerprint': plan_meta['plan_fingerprint'],
        'total_units': plan_meta['total_units'],
        'segment_id': segment_id,
        'segment_start_cursor': segment_start_cursor,
        'num_gpus': num_gpus,
        'per_device_batch': per_device_batch,
        'grad_accum': grad_accum,
        'segment_chain': list(segment_chain or []),
        'stop_file': stop_file,
        'output_dir': output_dir,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


def write_segment_config(segment_dir: str, config: Dict[str, Any]) -> str:
    path = os.path.join(segment_dir, SEGMENT_CONFIG_FILENAME)
    atomic_write_json(path, config)
    return path


def load_segment_config(config_value: Optional[str] = None) -> Dict[str, Any]:
    """Load the segment config from inline JSON or a file path.

    Inline JSON (``SWIFT_ELASTIC_RELAY_CONFIG='{"...": ...}'``) is preferred when every
    job runs in a fresh container where no file survives from the control plane.
    """
    config_value = config_value or os.environ.get(CONFIG_ENV)
    if not config_value:
        raise RuntimeError(f'Elastic-relay config not found. Set the `{CONFIG_ENV}` environment variable '
                           'to the segment config JSON (or a path to it) produced by the launcher.')
    config_value = config_value.strip()
    if config_value.startswith('{'):
        return json.loads(config_value)
    if not os.path.exists(config_value):
        raise FileNotFoundError(f'Elastic-relay config does not exist: {config_value}')
    return read_json(config_value)


# --------------------------------------------------------------------------------------
# platform state (written next to every checkpoint)
# --------------------------------------------------------------------------------------
def state_path_for_checkpoint(output_dir: str, global_step: int) -> str:
    return os.path.join(output_dir, f'checkpoint-{global_step}', STATE_FILENAME)


def parse_checkpoint_step(checkpoint_dir: str) -> Optional[int]:
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    if not name.startswith('checkpoint-'):
        return None
    try:
        return int(name[len('checkpoint-'):])
    except ValueError:
        return None


def read_platform_state(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    return read_json(os.path.join(checkpoint_dir, STATE_FILENAME))


def default_stop_file(output_dir: str) -> str:
    return os.path.join(output_dir, STOP_FILENAME)


def iter_checkpoint_dirs(search_dirs: List[str]) -> List[str]:
    """Yield every ``checkpoint-*`` directory found under the given roots."""
    found = []
    seen = set()
    for root in search_dirs:
        if not root or not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(root, '**', 'checkpoint-*'), recursive=True):
            if not os.path.isdir(path):
                continue
            if parse_checkpoint_step(path) is None:
                continue
            real = os.path.abspath(path)
            if real not in seen:
                seen.add(real)
                found.append(real)
    return found


def discover_committed_states(search_dirs: List[str]) -> List[Dict[str, Any]]:
    """Return all committed states, sorted ascending by ``committed_cursor``.

    Ties are broken by save time (and then by checkpoint step) so that the caller always
    gets the most advanced durable progress.
    """
    states = []
    for ckpt_dir in iter_checkpoint_dirs(search_dirs):
        state = read_platform_state(ckpt_dir)
        if state is None or not state.get('committed'):
            continue
        state = dict(state)
        state['checkpoint_path'] = ckpt_dir
        state['checkpoint_step'] = parse_checkpoint_step(ckpt_dir)
        states.append(state)
    states.sort(key=lambda s: (s.get('committed_cursor', -1), s.get('saved_at', ''), s.get('checkpoint_step', -1)))
    return states


def find_latest_committed_state(search_dirs: List[str]) -> Optional[Dict[str, Any]]:
    states = discover_committed_states(search_dirs)
    return states[-1] if states else None


def load_state_file(state_path: str) -> Dict[str, Any]:
    """Load a committed state either from a state json file or a checkpoint dir."""
    if os.path.isdir(state_path):
        state_path = os.path.join(state_path, STATE_FILENAME)
    if not os.path.exists(state_path):
        raise FileNotFoundError(f'Platform state not found: {state_path}')
    state = read_json(state_path)
    if not state.get('committed'):
        raise RuntimeError(f'Platform state is not committed: {state_path}')
    if 'checkpoint_path' not in state:
        state['checkpoint_path'] = os.path.dirname(os.path.abspath(state_path))
    return state
