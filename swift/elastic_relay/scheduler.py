# Copyright (c) ModelScope Contributors. All rights reserved.
"""Orchestration for one elastic-relay switch.

A switch is: read committed cursor -> plan remaining -> slice+pad -> validate -> emit
(or run) the ``swift sft`` command for the next segment.
"""
import os
import subprocess
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

from . import launcher
from .calculator import compute_first_segment, compute_next_segment
from .constants import RUNS_DIRNAME, SLICES_DIRNAME, STOP_FILENAME
from .plan_builder import load_plan_meta, verify_plan
from .slicer import slice_remaining
from .state import (build_segment_config, find_latest_committed_state, load_state_file, write_segment_config)
from .validator import validate

logger = get_logger()


def _segment_dir(workdir: str, segment_id: str) -> str:
    return os.path.join(workdir, RUNS_DIRNAME, segment_id)


def _slice_path(workdir: str, segment_id: str) -> str:
    return os.path.join(workdir, SLICES_DIRNAME, f'{segment_id}.jsonl')


def _next_segment_id(state: Optional[Dict[str, Any]]) -> str:
    chain = (state or {}).get('segment_chain') or []
    current = (state or {}).get('segment_id')
    # `state['segment_chain']` excludes the segment that produced the state, so count it too.
    completed = len(chain) + (1 if current and current not in chain else 0)
    return f'seg{completed + 1:02d}'


def resolve_state(workdir: str,
                  state: Optional[Dict[str, Any]] = None,
                  state_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Resolve the latest committed state from an explicit value, a path, or by discovery."""
    if state is not None:
        return state
    if state_path:
        return load_state_file(state_path)
    return find_latest_committed_state([os.path.join(workdir, RUNS_DIRNAME)])


def prepare_next_segment(workdir: str,
                         plan_meta: Dict[str, Any],
                         *,
                         num_gpus: int,
                         per_device_batch: int,
                         grad_accum: int,
                         base_model: str,
                         state: Optional[Dict[str, Any]] = None,
                         state_path: Optional[str] = None,
                         learning_rate: Optional[float] = None,
                         lr_strategy: str = 'sqrt',
                         save_steps: int = launcher.DEFAULT_SAVE_STEPS,
                         logging_steps: int = launcher.DEFAULT_LOGGING_STEPS,
                         save_total_limit: Optional[int] = None,
                         loss_type: Optional[str] = None,
                         swift_cmd: str = 'swift sft',
                         visible_devices: Optional[str] = None,
                         extra_args: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Prepare (but do not launch) the next segment. Returns ``None`` when training is DONE."""
    verify_plan(plan_meta)
    state = resolve_state(workdir, state, state_path)

    is_first = state is None
    if is_first:
        if learning_rate is None:
            raise ValueError('learning_rate is required for the first segment.')
        segment = compute_first_segment(plan_meta['total_units'], num_gpus, per_device_batch, grad_accum,
                                        learning_rate)
        segment_chain: List[str] = []
    else:
        segment = compute_next_segment(state, num_gpus, per_device_batch, grad_accum, lr_strategy=lr_strategy)
        segment_chain = list(state.get('segment_chain') or [])

    if segment is None:
        logger.info('elastic_relay: the training plan is fully consumed (DONE).')
        return None

    segment_id = _next_segment_id(state)
    output_dir = _segment_dir(workdir, segment_id)
    os.makedirs(output_dir, exist_ok=True)
    stop_file = os.path.join(output_dir, STOP_FILENAME)

    slice_path = _slice_path(workdir, segment_id)
    slice_meta = slice_remaining(plan_meta['plan_path'], segment['start_cursor'], segment['end_cursor'],
                                 segment['pad_count'], slice_path)

    ok, errors, warns = validate(segment, slice_path, plan_meta, state, num_gpus=num_gpus)
    for w in warns:
        logger.warning(f'elastic_relay[{segment_id}]: {w}')
    if not ok:
        raise RuntimeError(f'elastic_relay[{segment_id}] validation failed: {errors}')

    config = build_segment_config(
        plan_meta,
        segment_id,
        segment['start_cursor'],
        num_gpus,
        per_device_batch,
        grad_accum,
        segment_chain=segment_chain + ([state['segment_id']] if state and state.get('segment_id') else []),
        stop_file=stop_file,
        output_dir=output_dir)
    config_path = write_segment_config(output_dir, config)

    # Guarantee at least one save so the cursor advances even for tiny segments.
    effective_save_steps = max(1, min(save_steps, segment['estimated_steps']))
    if effective_save_steps != save_steps:
        logger.info(f'elastic_relay[{segment_id}]: save_steps adjusted {save_steps} -> '
                    f'{effective_save_steps} (estimated_steps={segment["estimated_steps"]}).')

    command = launcher.build_command(
        segment,
        slice_path,
        output_dir,
        num_gpus,
        per_device_batch,
        grad_accum,
        base_model,
        config_path,
        learning_rate=segment['new_lr'],
        save_steps=effective_save_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        loss_type=loss_type,
        swift_cmd=swift_cmd,
        visible_devices=visible_devices,
        extra_args=extra_args)

    plan = {
        'segment': segment,
        'segment_id': segment_id,
        'segment_chain': config['segment_chain'],
        'output_dir': output_dir,
        'slice_path': slice_path,
        'slice_meta': slice_meta,
        'config_path': config_path,
        'stop_file': stop_file,
        'command': command,
        'is_first': is_first,
    }
    return plan


def launch(plan: Dict[str, Any]) -> int:
    """Run a prepared segment synchronously and return the process exit code."""
    command = plan['command']
    logger.info(f"elastic_relay[{plan['segment_id']}]: launching\n{command['display']}")
    return subprocess.call(command['argv'], env=command['env'])


def switch(workdir: str,
           plan_meta: Dict[str, Any],
           *,
           num_gpus: int,
           per_device_batch: int,
           grad_accum: int,
           base_model: str,
           execute: bool = False,
           **kwargs) -> Optional[Dict[str, Any]]:
    """Prepare the next segment and optionally launch it."""
    plan = prepare_next_segment(
        workdir,
        plan_meta,
        num_gpus=num_gpus,
        per_device_batch=per_device_batch,
        grad_accum=grad_accum,
        base_model=base_model,
        **kwargs)
    if plan is None:
        return None
    if execute:
        plan['returncode'] = launch(plan)
    return plan


def status(workdir: str, plan_meta_path: Optional[str] = None) -> Dict[str, Any]:
    """Compute the current progress from the latest committed checkpoint."""
    plan_meta = load_plan_meta(plan_meta_path) if plan_meta_path else None
    state = find_latest_committed_state([os.path.join(workdir, RUNS_DIRNAME)])
    total = int(plan_meta['total_units']) if plan_meta else (int(state['total_units']) if state else 0)
    cursor = int(state['committed_cursor']) if state else 0
    return {
        'committed_cursor': cursor,
        'total_units': total,
        'remaining_units': total - cursor,
        'done': total > 0 and cursor >= total,
        'state': state,
    }
