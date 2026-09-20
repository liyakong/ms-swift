# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pre-flight checks before launching an elastic-relay segment."""
import os
from typing import Any, Dict, List, Optional, Tuple

from swift.utils import get_logger

from .slicer import count_rows

logger = get_logger()

_WEIGHT_SUFFIXES = ('.safetensors', '.bin', '.pt', '.pth')


def _checkpoint_is_complete(checkpoint_dir: str) -> Tuple[bool, str]:
    if not os.path.isdir(checkpoint_dir):
        return False, f'checkpoint directory does not exist: {checkpoint_dir}'
    has_config = any(
        os.path.exists(os.path.join(checkpoint_dir, name))
        for name in ('config.json', 'adapter_config.json', 'default'))
    has_weight = any(name.endswith(_WEIGHT_SUFFIXES) for name in os.listdir(checkpoint_dir))
    if not has_config or not has_weight:
        return False, f'checkpoint looks incomplete (config={has_config}, weight={has_weight}): {checkpoint_dir}'
    return True, ''


def validate(segment: Dict[str, Any],
             dataset_path: str,
             plan_meta: Dict[str, Any],
             state: Optional[Dict[str, Any]],
             *,
             num_gpus: Optional[int] = None) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warns: List[str] = []

    checkpoint = segment.get('checkpoint_path')
    if checkpoint:
        ok, msg = _checkpoint_is_complete(checkpoint)
        if not ok:
            errors.append(msg)
        if state is not None and not state.get('committed'):
            errors.append(f'checkpoint is not committed: {checkpoint}')

    if state is not None:
        if state.get('plan_fingerprint') != plan_meta.get('plan_fingerprint'):
            errors.append('plan fingerprint mismatch between platform_state and plan_meta; '
                          'the training plan was modified.')
        if int(state.get('total_units', -1)) != int(plan_meta.get('total_units', -2)):
            errors.append('total_units mismatch between platform_state and plan_meta.')

    if not os.path.exists(dataset_path):
        errors.append(f'sliced dataset does not exist: {dataset_path}')
    else:
        rows = count_rows(dataset_path)
        expected = int(segment['remaining_units']) + int(segment['pad_count'])
        if rows != expected:
            errors.append(f'sliced dataset has {rows} rows but expected {expected}.')

    if int(segment.get('estimated_steps', 0)) < 5:
        warns.append(f"this segment is only ~{segment.get('estimated_steps')} steps; the switch cost may "
                     'exceed the speedup.')
    new_lr = segment.get('new_lr')
    if new_lr is not None and new_lr > 1e-3:
        warns.append(f'new_lr={new_lr} looks large; verify it is intended.')

    pad = int(segment.get('pad_count', 0))
    if num_gpus and pad and pad >= num_gpus:
        warns.append(f'pad_count={pad} >= num_gpus={num_gpus}: with DDP strided sampling some ranks may '
                     'receive an all-padding micro-batch. The `elastic_cross_entropy` loss (registered '
                     'by the plugin and added by the launcher) keeps that numerically safe.')

    return (len(errors) == 0), errors, warns
