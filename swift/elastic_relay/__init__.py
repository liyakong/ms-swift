# Copyright (c) ModelScope Contributors. All rights reserved.
"""Elastic relay training (data-level checkpoint resume across changing GPU counts).

Public API::

    from swift.elastic_relay import (
        build_train_plan, load_plan_meta,
        prepare_next_segment, switch, status,
    )

The feature freezes a training plan, tracks a ``committed_cursor`` in a
``platform_state.json`` written next to every checkpoint, and slices the remaining
plan for each new segment. It guarantees no duplicated and no omitted training data
at the cost of not being bit-wise identical to an uninterrupted run.

Scope: SFT only (LoRA / full, DDP / DeepSpeed, local json/jsonl, optional multimodal).
"""
from .calculator import compute_first_segment, compute_next_segment, plan_segment
from .constants import (CONFIG_ENV, ELASTIC_RELAY_VERSION, LOSS_NAME, STATE_CALLBACK_NAME, STOP_CALLBACK_NAME)
from .plan_builder import (build_train_plan, dataset_fingerprint, load_plan_meta, load_raw_samples, verify_plan)
from .scheduler import launch, prepare_next_segment, resolve_state, status, switch
from .state import find_latest_committed_state, load_segment_config, load_state_file
from .validator import validate

__all__ = [
    'ELASTIC_RELAY_VERSION',
    'CONFIG_ENV',
    'STATE_CALLBACK_NAME',
    'STOP_CALLBACK_NAME',
    'LOSS_NAME',
    'build_train_plan',
    'load_plan_meta',
    'verify_plan',
    'dataset_fingerprint',
    'load_raw_samples',
    'plan_segment',
    'compute_first_segment',
    'compute_next_segment',
    'prepare_next_segment',
    'resolve_state',
    'switch',
    'launch',
    'status',
    'validate',
    'find_latest_committed_state',
    'load_state_file',
    'load_segment_config',
]
