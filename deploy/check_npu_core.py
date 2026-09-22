#!/usr/bin/env python
"""Build/runtime check: the NPU stack must match the verified environment.

Reads `/constraints-tested.txt` (generated from the verified container) and compares the
core packages to what is actually installed. Metadata only: never imports torch/swift.
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CORE = (
    'torch',
    'torch-npu',
    'torchvision',
    'torchaudio',
    'triton',
    'triton-ascend',
    'vllm',
    'vllm-ascend',
)


def main() -> int:
    tested = {}
    constraints = Path('/constraints-tested.txt')
    if not constraints.exists():
        print('[core-check] /constraints-tested.txt not found; skipped')
        return 0
    for line in constraints.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            name, ver = line.split('==', 1)
            tested[name] = ver

    for name in CORE:
        try:
            actual = version(name)
        except PackageNotFoundError:
            print(f'[core-check] {name}: not installed (skipped)')
            continue
        expected = tested.get(name)
        if expected is not None and actual != expected:
            raise RuntimeError(f'{name}: current={actual}, verified={expected}. '
                               'Check the base image; do not upgrade/downgrade the NPU stack.')
        print(f'[core-check] {name}=={actual}')

    try:
        triton_ascend = version('triton-ascend')
    except PackageNotFoundError:
        raise RuntimeError('triton-ascend is missing; this image requires triton-ascend==3.2.2')
    if triton_ascend != '3.2.2':
        raise RuntimeError(f'This image is based on triton-ascend==3.2.2, found {triton_ascend}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
