#!/usr/bin/env python
"""Build-time acceptance: FLA version, key packages, and required files exist.

Runs at image build time (no NPU init). Reads FLA_VERSION / VLLM_ASCEND_CUSTOM_VENDOR /
ARCH from the environment.
"""
import os
import sys
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    expected = os.environ['FLA_VERSION']
    for name in ('flash-linear-attention', 'fla-core'):
        actual = version(name)
        if actual != expected:
            raise RuntimeError(f'{name}: {actual} != {expected}')
        print(f'[build-check] {name}=={actual}')

    for name in ('ms-swift', 'transformers', 'einops'):
        print(f'[build-check] {name}=={version(name)}')

    vendor = Path(os.environ['VLLM_ASCEND_CUSTOM_VENDOR'])
    arch = os.environ['ARCH']
    required = [
        vendor / 'op_api/lib/libcust_opapi.so',
        vendor / f'op_proto/lib/linux/{arch}/libcust_opsproto_rt2.0.so',
        vendor / 'op_impl/ai_core/tbe/kernel/ascend910b/fused_gdn_gating',
        Path('/swift/elastic_relay/plugin.py'),
        Path('/swift/elastic_relay/runner.py'),
    ]
    for path in required:
        if not path.exists():
            raise RuntimeError(f'missing build artifact or source file: {path}')
        print(f'[build-check] exists: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
