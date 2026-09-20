# Copyright (c) ModelScope Contributors. All rights reserved.
"""Command-line entry point for elastic relay training.

Examples::

    python -m swift.elastic_relay.cli build-plan \
        --raw /data/train.jsonl --workdir /work/run1 --epochs 2 --seed 42

    python -m swift.elastic_relay.cli status --workdir /work/run1

    python -m swift.elastic_relay.cli next \
        --workdir /work/run1 --num-gpus 8 --per-device-batch 2 --grad-accum 1 \
        --base-model Qwen/Qwen3-8B --learning-rate 1e-4 --execute
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from .constants import PLAN_META_FILENAME
from .plan_builder import build_train_plan, load_plan_meta
from .scheduler import prepare_next_segment, status


def _default_plan_meta(workdir: str) -> str:
    return os.path.join(workdir, 'plan', PLAN_META_FILENAME)


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('swift.elastic_relay', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    p_build = sub.add_parser('build-plan', help='Freeze the raw dataset into a training plan.')
    p_build.add_argument('--raw', required=True, help='Path to the raw train.jsonl.')
    p_build.add_argument('--workdir', required=True)
    p_build.add_argument('--epochs', type=int, default=1, dest='num_train_epochs')
    p_build.add_argument('--seed', type=int, default=42)
    p_build.add_argument('--plan-dir', default=None, help='Defaults to <workdir>/plan.')

    p_next = sub.add_parser('next', help='Prepare (and optionally launch) the next segment.')
    p_next.add_argument('--workdir', required=True)
    p_next.add_argument('--plan-meta', default=None)
    p_next.add_argument('--num-gpus', type=int, required=True)
    p_next.add_argument('--per-device-batch', type=int, required=True)
    p_next.add_argument('--grad-accum', type=int, required=True)
    p_next.add_argument('--base-model', required=True)
    p_next.add_argument('--learning-rate', type=float, default=None)
    p_next.add_argument('--lr-strategy', choices=['constant', 'sqrt', 'linear'], default='sqrt')
    p_next.add_argument('--state-path', default=None)
    p_next.add_argument('--save-steps', type=int, default=100)
    p_next.add_argument('--logging-steps', type=int, default=5)
    p_next.add_argument('--save-total-limit', type=int, default=None)
    p_next.add_argument('--loss-type', default=None)
    p_next.add_argument('--swift-cmd', default='swift sft')
    p_next.add_argument('--execute', action='store_true')
    p_next.add_argument('extra_args', nargs=argparse.REMAINDER, help='Extra args forwarded to swift sft.')

    p_status = sub.add_parser('status', help='Show current cursor progress.')
    p_status.add_argument('--workdir', required=True)
    p_status.add_argument('--plan-meta', default=None)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == 'build-plan':
        plan_dir = args.plan_dir or os.path.join(args.workdir, 'plan')
        meta = build_train_plan(args.raw, plan_dir, num_train_epochs=args.num_train_epochs, seed=args.seed)
        _print(meta)
        return 0

    if args.command == 'status':
        plan_meta_path = args.plan_meta or _default_plan_meta(args.workdir)
        plan_meta_path = plan_meta_path if os.path.exists(plan_meta_path) else None
        _print(status(args.workdir, plan_meta_path))
        return 0

    if args.command == 'next':
        plan_meta_path = args.plan_meta or _default_plan_meta(args.workdir)
        plan_meta = load_plan_meta(plan_meta_path)
        extra_args = [a for a in (args.extra_args or []) if a != '--']
        plan = prepare_next_segment(
            args.workdir,
            plan_meta,
            num_gpus=args.num_gpus,
            per_device_batch=args.per_device_batch,
            grad_accum=args.grad_accum,
            base_model=args.base_model,
            state_path=args.state_path,
            learning_rate=args.learning_rate,
            lr_strategy=args.lr_strategy,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            save_total_limit=args.save_total_limit,
            loss_type=args.loss_type,
            swift_cmd=args.swift_cmd,
            extra_args=extra_args)
        if plan is None:
            _print({'done': True})
            return 0

        if args.execute:
            from .scheduler import launch
            plan['returncode'] = launch(plan)

        summary: Dict[str, Any] = {
            'segment_id': plan['segment_id'],
            'segment': plan['segment'],
            'output_dir': plan['output_dir'],
            'slice_path': plan['slice_path'],
            'config_path': plan['config_path'],
            'stop_file': plan['stop_file'],
            'command': plan['command']['display'],
        }
        if 'returncode' in plan:
            summary['returncode'] = plan['returncode']
        _print(summary)
        return 0

    return 1


if __name__ == '__main__':
    sys.exit(main())
