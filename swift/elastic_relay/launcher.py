# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build the ``swift sft`` command (and environment) for one elastic-relay segment."""
import os
import shlex
from typing import Any, Dict, List, Optional

from swift.utils import get_logger

from .constants import CONFIG_ENV, LOSS_NAME, visible_device_env

logger = get_logger()

DEFAULT_SAVE_STEPS = 100
DEFAULT_LOGGING_STEPS = 5
DEFAULT_LOSS_TYPE_FOR_PADDING = LOSS_NAME


def _plugin_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plugin.py')


def build_command(segment: Dict[str, Any],
                  dataset_path: str,
                  output_dir: str,
                  num_gpus: int,
                  per_device_batch: int,
                  grad_accum: int,
                  base_model: str,
                  config_path: str,
                  *,
                  learning_rate: Optional[float] = None,
                  save_steps: int = DEFAULT_SAVE_STEPS,
                  logging_steps: int = DEFAULT_LOGGING_STEPS,
                  save_total_limit: Optional[int] = None,
                  loss_type: Optional[str] = None,
                  swift_cmd: str = 'swift sft',
                  visible_devices: Optional[str] = None,
                  extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return ``{'argv', 'env', 'display'}`` for a segment.

    ``argparse``-style booleans are passed explicitly (e.g. ``--foo true``) to avoid the
    ambiguity of swift's optional-bool flags.
    """
    argv: List[str] = shlex.split(swift_cmd)

    checkpoint_path = segment.get('checkpoint_path')
    if checkpoint_path:
        # Load weights only; start from global_step 0 and do NOT skip data inside the framework.
        argv += [
            '--resume_from_checkpoint', checkpoint_path,
            '--resume_only_model', 'true',
            '--ignore_data_skip', 'true',
        ]
    argv += ['--model', base_model]
    argv += ['--dataset', dataset_path]

    argv += ['--num_train_epochs', '1']
    argv += ['--dataset_shuffle', 'false']
    argv += ['--train_dataloader_shuffle', 'false']
    argv += ['--group_by_length', 'false']
    argv += ['--per_device_train_batch_size', str(per_device_batch)]
    argv += ['--gradient_accumulation_steps', str(grad_accum)]

    if learning_rate is not None:
        argv += ['--learning_rate', repr(learning_rate)]
    argv += ['--lr_scheduler_type', 'constant_with_warmup']
    argv += ['--warmup_steps', str(segment.get('warmup_steps', 0))]

    argv += ['--output_dir', output_dir]
    argv += ['--save_strategy', 'steps', '--save_steps', str(save_steps)]
    argv += ['--logging_steps', str(logging_steps)]
    if save_total_limit is not None:
        argv += ['--save_total_limit', str(save_total_limit)]
    # Every segment is padded to a whole number of global batches, so dropping the last
    # partial batch is a no-op and guarantees equal optimizer-step counts across ranks.
    argv += ['--dataloader_drop_last', 'true']
    argv += ['--load_args', 'false']

    if segment.get('pad_count'):
        # The default SFT path lets the model compute its own mean loss, which is NaN for
        # an all-padding micro-batch. `elastic_cross_entropy` (registered by the plugin)
        # uses reduction='none' and a clamped denominator, so it is DDP-safe.
        argv += ['--loss_type', loss_type or DEFAULT_LOSS_TYPE_FOR_PADDING]

    argv += ['--external_plugins', _plugin_path()]
    argv += ['--callbacks', 'elastic_stop', 'elastic_state']

    if extra_args:
        argv += [str(a) for a in extra_args]

    env = dict(os.environ)
    # Never hardcode a vendor variable: keep the platform's device visibility (Ascend/CUDA/...)
    # unless the caller explicitly asks to pin a list.
    if visible_devices is not None:
        name, _ = visible_device_env()
        env[name or 'CUDA_VISIBLE_DEVICES'] = visible_devices
    env[CONFIG_ENV] = os.path.abspath(config_path)

    display = ' '.join(shlex.quote(part) for part in argv)
    return {'argv': argv, 'env': env, 'display': display}
