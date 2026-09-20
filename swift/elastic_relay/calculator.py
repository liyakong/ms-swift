# Copyright (c) ModelScope Contributors. All rights reserved.
"""Convert a committed cursor + new hardware shape into a concrete segment plan."""
import math
from typing import Any, Dict, Optional

from swift.utils import get_logger

logger = get_logger()

MIN_WARMUP_STEPS = 10
MAX_WARMUP_STEPS = 100
LR_SCALE_CAP = 3.0
WARMUP_STEP_FRACTION = 20


def _scale_lr(old_lr: float, old_gb: int, new_gb: int, start_cursor: int, total_units: int,
              strategy: str) -> float:
    if not old_lr or not old_gb:
        return old_lr
    ratio = new_gb / old_gb
    if strategy == 'constant':
        scale = 1.0
    elif strategy == 'linear':
        scale = ratio
    elif strategy == 'sqrt':
        scale = math.sqrt(ratio)
    else:
        raise ValueError(f'Unknown lr_strategy: {strategy!r} (expected constant/sqrt/linear)')
    progress = start_cursor / total_units if total_units else 0.0
    # Be progressively more conservative as training approaches the end.
    scale = 1.0 + (scale - 1.0) * (1.0 - progress)
    return min(old_lr * scale, old_lr * LR_SCALE_CAP)


def plan_segment(start_cursor: int,
                 total_units: int,
                 new_global_batch: int,
                 *,
                 old_global_batch: Optional[int] = None,
                 current_lr: Optional[float] = None,
                 lr_strategy: str = 'sqrt',
                 checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Build the plan for one segment starting at ``start_cursor``.

    Returns ``None`` when the plan has already been fully consumed.
    """
    if new_global_batch <= 0:
        raise ValueError(f'new_global_batch must be positive, got {new_global_batch}')
    remaining = total_units - start_cursor
    if remaining <= 0:
        return None

    padded = math.ceil(remaining / new_global_batch) * new_global_batch
    pad_count = padded - remaining
    estimated_steps = padded // new_global_batch
    warmup_steps = max(MIN_WARMUP_STEPS, min(MAX_WARMUP_STEPS, estimated_steps // WARMUP_STEP_FRACTION))

    new_lr = current_lr
    if current_lr is not None and old_global_batch:
        new_lr = _scale_lr(current_lr, old_global_batch, new_global_batch, start_cursor, total_units, lr_strategy)

    return {
        'start_cursor': start_cursor,
        'end_cursor': total_units,
        'remaining_units': remaining,
        'new_global_batch': new_global_batch,
        'pad_count': pad_count,
        'estimated_steps': estimated_steps,
        'new_lr': round(new_lr, 8) if new_lr is not None else None,
        'warmup_steps': warmup_steps,
        'checkpoint_path': checkpoint_path,
    }


def compute_next_segment(state: Dict[str, Any],
                         new_num_gpus: int,
                         new_per_device_batch: int,
                         new_grad_accum: int,
                         lr_strategy: str = 'sqrt') -> Optional[Dict[str, Any]]:
    """Plan the segment that follows the given committed state."""
    cursor = int(state['committed_cursor'])
    total = int(state['total_units'])
    new_gb = int(new_num_gpus) * int(new_per_device_batch) * int(new_grad_accum)
    return plan_segment(
        cursor,
        total,
        new_gb,
        old_global_batch=int(state.get('segment_global_batch') or 0) or None,
        current_lr=state.get('current_lr'),
        lr_strategy=lr_strategy,
        checkpoint_path=state.get('checkpoint_path'),
    )


def compute_first_segment(total_units: int,
                          new_num_gpus: int,
                          new_per_device_batch: int,
                          new_grad_accum: int,
                          learning_rate: float) -> Dict[str, Any]:
    new_gb = int(new_num_gpus) * int(new_per_device_batch) * int(new_grad_accum)
    return plan_segment(0, total_units, new_gb, current_lr=learning_rate, checkpoint_path=None)
