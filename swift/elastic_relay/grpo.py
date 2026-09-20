# Copyright (c) ModelScope Contributors. All rights reserved.
"""GRPO-specific accounting for elastic relay (Phase 1: fixed card count).

GRPO consumes *prompts* in **generation rounds**, not samples per optimizer step. With
``dataset_shuffle=false`` the sampler iterates the (sliced) dataset sequentially, so the
unique-prompt order equals our plan order regardless of how many cards are used. The
progress unit is therefore a whole generation round::

    prompts_per_round     P = generation_batch_size // num_generations
    optimizer steps/round S = steps_per_generation * num_iterations // gradient_accumulation_steps
    committed_prompts       = start_cursor + (global_step // S) * P

This module only depends on stdlib so it can be unit-tested without torch/trl.
"""
from typing import Any, Dict, Optional

# Defaults mirrored from swift/arguments/rlhf_args.py and args_mixin.py.
GRPO_DEFAULT_NUM_GENERATIONS = 8
GRPO_DEFAULT_NUM_ITERATIONS = 1
GRPO_DEFAULT_GRAD_ACCUM = 1


def resolve_generation_params(per_device_batch: int,
                              world_size: int,
                              num_generations: int = GRPO_DEFAULT_NUM_GENERATIONS,
                              num_iterations: int = GRPO_DEFAULT_NUM_ITERATIONS,
                              gradient_accumulation_steps: Optional[int] = None,
                              steps_per_generation: Optional[int] = None,
                              generation_batch_size: Optional[int] = None) -> Dict[str, Any]:
    """Replicate ``RolloutTrainerArgumentsMixin._init_generation_batch_params``.

    Raises ``ValueError`` on any configuration the framework itself would reject, so we
    fail before slicing instead of after.
    """
    if num_generations < 1:
        raise ValueError(f'num_generations must be >= 1, got {num_generations}')
    if num_iterations < 1:
        raise ValueError(f'num_iterations must be >= 1, got {num_iterations}')
    if world_size < 1:
        raise ValueError(f'world_size must be >= 1, got {world_size}')

    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = GRPO_DEFAULT_GRAD_ACCUM
    if gradient_accumulation_steps < 1:
        raise ValueError(f'gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}')

    global_batch_size = per_device_batch * world_size

    if generation_batch_size is None and steps_per_generation is None:
        steps_per_generation = gradient_accumulation_steps
        generation_batch_size = global_batch_size * steps_per_generation
    elif generation_batch_size is not None and steps_per_generation is None:
        if generation_batch_size % global_batch_size != 0:
            raise ValueError(f'generation_batch_size ({generation_batch_size}) must be divisible by the global '
                             f'batch size ({global_batch_size}).')
        steps_per_generation = generation_batch_size // global_batch_size
    elif generation_batch_size is None and steps_per_generation is not None:
        generation_batch_size = global_batch_size * steps_per_generation
    else:
        expected = global_batch_size * steps_per_generation
        if generation_batch_size != expected:
            raise ValueError(f'generation_batch_size ({generation_batch_size}) must equal per_device_train_batch_size'
                             f' * world_size * steps_per_generation = {expected}.')

    if steps_per_generation <= 0:
        raise ValueError(f'steps_per_generation must be > 0, got {steps_per_generation}')
    if num_generations > 1 and generation_batch_size % num_generations != 0:
        raise ValueError(f'generation_batch_size ({generation_batch_size}) must be evenly divisible by '
                         f'num_generations ({num_generations}).')

    prompts_per_round = generation_batch_size // num_generations
    micro_steps_per_round = steps_per_generation * num_iterations
    if micro_steps_per_round % gradient_accumulation_steps != 0:
        raise ValueError(f'steps_per_generation * num_iterations ({micro_steps_per_round}) must be divisible by '
                         f'gradient_accumulation_steps ({gradient_accumulation_steps}).')
    optimizer_steps_per_round = micro_steps_per_round // gradient_accumulation_steps

    return {
        'per_device_batch': per_device_batch,
        'world_size': world_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'num_generations': num_generations,
        'num_iterations': num_iterations,
        'global_batch_size': global_batch_size,
        'steps_per_generation': steps_per_generation,
        'generation_batch_size': generation_batch_size,
        'prompts_per_round': prompts_per_round,
        'optimizer_steps_per_round': optimizer_steps_per_round,
    }


def round_aligned_end(start_cursor: int, total_units: int, prompts_per_round: int) -> int:
    """Largest ``end <= total_units`` such that ``end - start_cursor`` is a whole number of rounds."""
    if prompts_per_round <= 0:
        raise ValueError(f'prompts_per_round must be > 0, got {prompts_per_round}')
    remaining = total_units - start_cursor
    rounds = remaining // prompts_per_round
    return start_cursor + rounds * prompts_per_round


def completed_rounds(global_step: int, optimizer_steps_per_round: int) -> int:
    if optimizer_steps_per_round <= 0:
        raise ValueError(f'optimizer_steps_per_round must be > 0, got {optimizer_steps_per_round}')
    return max(global_step, 0) // optimizer_steps_per_round


def committed_prompts(start_cursor: int,
                      global_step: int,
                      optimizer_steps_per_round: int,
                      prompts_per_round: int,
                      total_units: Optional[int] = None) -> int:
    committed = start_cursor + completed_rounds(global_step, optimizer_steps_per_round) * prompts_per_round
    if total_units is not None:
        committed = min(committed, total_units)
    return committed


def is_round_boundary(global_step: int, optimizer_steps_per_round: int) -> bool:
    if optimizer_steps_per_round <= 0:
        return False
    return global_step % optimizer_steps_per_round == 0
