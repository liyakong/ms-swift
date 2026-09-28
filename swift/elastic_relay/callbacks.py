# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trainer callbacks for elastic relay training.

Both callbacks are registered by ``swift/elastic_relay/plugin.py`` (loaded through
``--external_plugins``) and are therefore referenced by name::

    --callbacks elastic_stop elastic_state

They receive only ``(args, trainer)`` from ``SwiftMixin._get_callbacks``, so all
per-segment parameters are read from the JSON file pointed at by the
``SWIFT_ELASTIC_RELAY_CONFIG`` environment variable.
"""
import os
import time
from typing import TYPE_CHECKING, Any, Dict

import torch
from transformers import TrainerControl, TrainerState

from swift.callbacks.base import TrainerCallback
from swift.utils import ShutdownManager, get_device, get_logger

from .constants import STATE_FILENAME
from .grpo import is_round_boundary
from .state import atomic_write_json, default_stop_file, load_segment_config, scrub_external_plugins

if TYPE_CHECKING:
    from swift.trainers import Trainer, TrainingArguments

logger = get_logger()


def _resolve_global_batch(args: 'TrainingArguments', config: Dict[str, Any]) -> int:
    # Prefer the planner's resolved values: DeepSpeed may override gradient accumulation,
    # so the runtime `args.gradient_accumulation_steps` can differ from the effective value.
    num_gpus = config.get('num_gpus')
    per_device_batch = config.get('per_device_batch')
    grad_accum = config.get('grad_accum')
    if num_gpus and per_device_batch and grad_accum:
        return int(num_gpus) * int(per_device_batch) * int(grad_accum)
    world_size = getattr(args, 'world_size', None) or num_gpus or 1
    return args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size


def _latest_learning_rate(args: 'TrainingArguments', state: TrainerState) -> float:
    for record in reversed(state.log_history):
        if 'learning_rate' in record:
            return record['learning_rate']
    return args.learning_rate


class ElasticStateCallback(TrainerCallback):
    """Writes ``platform_state.json`` (with ``committed_cursor``) whenever a checkpoint is saved.

    * SFT:  ``committed_cursor = start + global_step * global_batch`` (unit: sample).
    * GRPO: ``committed_cursor = start + (global_step // optimizer_steps_per_round) *
      prompts_per_round`` (unit: prompt). Only generation-round boundaries are committed;
      mid-round checkpoints are ignored so a resume never lands weights ahead of the cursor.
    """

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)
        self.config = load_segment_config()
        self.mode = self.config.get('mode', 'sft')
        self.start_cursor = int(self.config['segment_start_cursor'])
        self.total_units = int(self.config['total_units'])
        self.plan_fingerprint = self.config['plan_fingerprint']
        self.segment_id = self.config['segment_id']
        self.segment_chain = list(self.config.get('segment_chain') or [])
        self.prompts_per_round = int(self.config.get('prompts_per_round') or 0)
        self.opt_steps_per_round = int(self.config.get('optimizer_steps_per_round') or 0)
        if self.mode == 'grpo':
            self.global_batch = int(self.config.get('global_batch_size') or 0)
        else:
            self.global_batch = _resolve_global_batch(args, self.config)
            expected_gb = (int(self.config.get('num_gpus', 0)) * int(self.config.get('per_device_batch', 0))
                           * int(self.config.get('grad_accum', 0)))
            if expected_gb and expected_gb != self.global_batch:
                logger.warning(f'elastic_relay: runtime global_batch is {self.global_batch}, '
                               f'but the scheduler planned {expected_gb}. Using the runtime value.')

    @property
    def is_world_process_zero(self) -> bool:
        return self.args.process_index == 0

    def _committed(self, global_step: int) -> int:
        if self.mode == 'grpo':
            committed = (self.start_cursor
                         + (global_step // self.opt_steps_per_round) * self.prompts_per_round)
        else:
            committed = self.start_cursor + global_step * self.global_batch
        return min(committed, self.total_units)

    def on_step_end(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        # Always persist the final step, independently of `save_steps` / `save_strategy`.
        max_steps = getattr(state, 'max_steps', None)
        if max_steps is not None and max_steps > 0 and state.global_step >= max_steps:
            logger.info(f'elastic_relay[{self.segment_id}]: reached the last step '
                        f'({state.global_step}/{max_steps}); forcing a final checkpoint save.')
            control.should_save = True
        return control

    def on_save(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        if not self.is_world_process_zero:
            return control
        if self.mode == 'grpo' and not is_round_boundary(state.global_step, self.opt_steps_per_round):
            logger.warning(f'elastic_relay[{self.segment_id}]: checkpoint-{state.global_step} is mid-generation-round '
                           f'(boundary every {self.opt_steps_per_round} optimizer steps); not committing a cursor.')
            return control

        ckpt_dir = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        if not os.path.isdir(ckpt_dir):
            logger.warning(f'elastic_relay: on_save fired but {ckpt_dir} does not exist; skipping state write.')
            return control

        committed = self._committed(state.global_step)
        payload = {
            'mode': self.mode,
            'cursor_unit': 'prompt' if self.mode == 'grpo' else 'sample',
            'plan_fingerprint': self.plan_fingerprint,
            'total_units': self.total_units,
            'segment_start_cursor': self.start_cursor,
            'segment_global_step': state.global_step,
            'committed_cursor': committed,
            'remaining_units': self.total_units - committed,
            'num_gpus': self.config.get('num_gpus'),
            'per_device_batch': self.config.get('per_device_batch', args.per_device_train_batch_size),
            'grad_accum': self.config.get('grad_accum', args.gradient_accumulation_steps),
            'current_lr': _latest_learning_rate(args, state),
            'checkpoint_path': ckpt_dir,
            'segment_id': self.segment_id,
            'segment_chain': self.segment_chain,
            # recipe (used to detect plan-defining changes across segments)
            'seed': self.config.get('seed'),
            'num_train_epochs': self.config.get('num_train_epochs'),
            'dataset_size': self.config.get('dataset_size'),
            'raw_dataset_fingerprint': self.config.get('raw_dataset_fingerprint'),
            'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'committed': True,
        }
        if self.mode == 'grpo':
            payload.update({
                'segment_global_batch': self.global_batch,
                'num_generations': self.config.get('num_generations'),
                'num_iterations': self.config.get('num_iterations'),
                'steps_per_generation': self.config.get('steps_per_generation'),
                'generation_batch_size': self.config.get('generation_batch_size'),
                'prompts_per_round': self.prompts_per_round,
                'optimizer_steps_per_round': self.opt_steps_per_round,
                # reference-model identity (must stay pinned across segments)
                'tuner_type': self.config.get('tuner_type'),
                'beta': self.config.get('beta'),
                'ref_model': self.config.get('ref_model'),
                'ref_adapters': self.config.get('ref_adapters'),
                'sync_ref_model': self.config.get('sync_ref_model'),
                # rollout backend / sampling (must stay consistent across segments)
                'use_vllm': self.config.get('use_vllm'),
                'vllm_mode': self.config.get('vllm_mode'),
                'temperature': self.config.get('temperature'),
                'top_p': self.config.get('top_p'),
                'top_k': self.config.get('top_k'),
                'max_completion_length': self.config.get('max_completion_length'),
            })
        else:
            payload['segment_global_batch'] = self.global_batch
        atomic_write_json(os.path.join(ckpt_dir, STATE_FILENAME), payload)
        # Keep checkpoints portable: blank the injected plugin path as soon as each checkpoint
        # is written (the runner re-injects it on resume, so training is unaffected).
        scrub_external_plugins([os.path.join(ckpt_dir, 'args.json'), os.path.join(args.output_dir, 'args.json')])
        cursor_unit = 'prompt' if self.mode == 'grpo' else 'sample'
        # `mode` is the *accounting* mode (DPO/PT reuse the SFT one); also show how this segment
        # was launched so the log is not mistaken for plain SFT. `pt`/`sft` are *routes*
        # (`swift pt` / `swift sft`), while `rlhf_type` only applies to the `rlhf` route.
        route = self.config.get('route') or self.mode
        rlhf_type = self.config.get('rlhf_type')
        rlhf_suffix = f', rlhf_type={rlhf_type}' if rlhf_type else ''
        if self.mode == 'grpo':
            detail = f'P={self.prompts_per_round}, OSR={self.opt_steps_per_round}, step={state.global_step}'
        else:
            detail = f'start={self.start_cursor} + step={state.global_step}xgb={self.global_batch}'
        logger.info(f'elastic_relay[{self.segment_id}]: committed_cursor={committed}/{self.total_units} '
                    f'(accounting={self.mode}/{cursor_unit}, route={route}{rlhf_suffix}, {detail})')
        return control


class ElasticStopCallback(TrainerCallback):
    """Soft stop on either the stop file or a termination signal.

    On the stop request we save at the next optimizer-step boundary and exit, regardless
    of `save_steps`. Signals are handled through the framework's ``ShutdownManager`` so a
    plain ``SIGTERM`` (e.g. from Kubernetes) also produces a checkpoint.
    """

    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        super().__init__(args, trainer)
        self.config = load_segment_config()
        self.mode = self.config.get('mode', 'sft')
        self.opt_steps_per_round = int(self.config.get('optimizer_steps_per_round') or 0)
        self.stop_file = self.config.get('stop_file') or default_stop_file(args.output_dir)
        self._pending_stop = False
        self._shutdown_manager = None
        try:
            manager = ShutdownManager(stop_file=self.stop_file)
            manager.register()
            self._shutdown_manager = manager
        except Exception as e:  # pragma: no cover - platform dependent
            logger.warning(f'elastic_relay: failed to register shutdown signal handlers: {e}')

    def _stop_requested(self) -> bool:
        if os.path.exists(self.stop_file):
            return True
        return bool(self._shutdown_manager is not None and self._shutdown_manager.should_shutdown())

    def on_step_end(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        local_req = 1 if self._stop_requested() else 0
        any_req = bool(local_req)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device_type = get_device()
            tensor = torch.tensor([local_req], dtype=torch.uint8, device=device_type)
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
            any_req = bool(int(tensor.item()))

        if any_req:
            # GRPO: defer the stop to a generation-round boundary so the committed cursor and
            # the saved weights stay aligned (a mid-round save cannot be committed).
            if self.mode == 'grpo' and not is_round_boundary(state.global_step, self.opt_steps_per_round):
                logger.info(f'elastic_relay: stop requested mid-generation-round at step {state.global_step}; '
                            f'deferring to the next boundary (every {self.opt_steps_per_round} steps).')
                return control
            logger.info('elastic_relay: stop requested (stop file or signal); '
                        'saving a checkpoint before exiting.')
            control.should_save = True
            self._pending_stop = True
        return control

    def on_save(self, args: 'TrainingArguments', state: TrainerState, control: TrainerControl, **kwargs):
        if self._pending_stop:
            control.should_training_stop = True
            self._pending_stop = False
            self._consume_stop_file()
        return control

    def _consume_stop_file(self) -> None:
        """Delete the stop file after the stop checkpoint is saved (rank 0 only).

        Best-effort one-shot semantics: the runner also clears any leftover at the next
        segment's startup, so a residual file can never stop a later segment.
        """
        if getattr(self.args, 'process_index', 0) != 0:
            return
        try:
            if os.path.exists(self.stop_file):
                os.remove(self.stop_file)
                logger.info(f'elastic_relay: removed the consumed stop file {self.stop_file}.')
        except OSError as e:  # pragma: no cover - best effort
            logger.warning(f'elastic_relay: failed to remove stop file {self.stop_file}: {e}')
