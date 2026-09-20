# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared constants for the elastic relay (data-level checkpoint resume) feature."""
import os

# Frozen scope: SFT only. Bump on any change to the plan/cursor format or public behavior.
ELASTIC_RELAY_VERSION = '1.0.0'

# Environment variable carrying the per-segment config (inline JSON or a path to it).
# The callbacks read it at construction time; the launcher sets it before `swift sft`.
CONFIG_ENV = 'SWIFT_ELASTIC_RELAY_CONFIG'

# Device-visibility environment variables, in priority order (NPU first, then GPU-like).
# We never hardcode a single vendor's variable so the same code works on Ascend/CUDA/etc.
VISIBLE_DEVICE_ENVS = (
    'ASCEND_RT_VISIBLE_DEVICES',
    'CUDA_VISIBLE_DEVICES',
    'HIP_VISIBLE_DEVICES',
    'XPU_VISIBLE_DEVICES',
    'MLU_VISIBLE_DEVICES',
    'NPU_VISIBLE_DEVICES',
)


def visible_device_env():
    """Return ``(name, value)`` of the device-visibility env var in use, or ``(None, None)``."""
    for name in VISIBLE_DEVICE_ENVS:
        value = os.environ.get(name)
        if value:
            return name, value
    return None, None


def count_visible_devices() -> int:
    value = visible_device_env()[1]
    if not value:
        return 0
    return len([x for x in value.split(',') if x.strip()])


# Names registered into `swift.callbacks.callbacks_map` / `swift.loss.loss_map` by `plugin.py`.
STATE_CALLBACK_NAME = 'elastic_state'
STOP_CALLBACK_NAME = 'elastic_stop'
LOSS_NAME = 'elastic_cross_entropy'

# Files written into the workdir / checkpoint dirs.
PLAN_FILENAME = 'train_plan.jsonl'
PLAN_META_FILENAME = 'plan_meta.json'
STATE_FILENAME = 'platform_state.json'
SLICE_META_FILENAME = 'slice_meta.json'
SEGMENT_CONFIG_FILENAME = 'elastic_segment.json'
STOP_FILENAME = 'STOP'

# Fallback content for a padding sample. Written as a user-only conversation so the
# template produces an all `-100` label row (no supervised tokens).
DUMMY_CONTENT = '<elastic-relay-pad>'

RUNS_DIRNAME = 'runs'
SLICES_DIRNAME = 'slices'
