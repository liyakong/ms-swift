#!/usr/bin/env python
"""Build/runtime check: the NPU stack must match the verified environment.

Reads `/constraints-npu.txt` (the NPU-stack allowlist, tracked in the repo) and compares the
core packages to what is actually installed. Metadata only: never imports torch/swift.

Also fails if a CUDA/GPU build slipped in (e.g. nvidia-cublas-cu12 pulled by a CUDA torch
wheel): this image must stay on the NPU (torch +cpu) stack.
"""
import re
import sys
from importlib.metadata import PackageNotFoundError, distributions, version
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

# CUDA runtime wheels that only appear when a CUDA torch build was installed.
CUDA_DIST_RE = re.compile(r'^nvidia-(cublas|cuda|cudnn|curand|cufft|cufile|cupti|cusolver'
                          r'|cusparse|cusparselt|nccl|nvjitlink|nvtx)')


def canon(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def main() -> int:
    tested = {}
    constraints = Path('/constraints-npu.txt')
    if not constraints.exists():
        print('[core-check] /constraints-npu.txt not found; skipped')
        return 0
    for line in constraints.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            name, ver = line.split('==', 1)
            tested[canon(name)] = ver.strip()

    for name in CORE:
        try:
            actual = version(name)
        except PackageNotFoundError:
            print(f'[core-check] {name}: not installed (skipped)')
            continue
        expected = tested.get(canon(name))
        if expected is not None and actual != expected:
            raise RuntimeError(f'{name}: current={actual}, verified={expected}. '
                               'Check the base image; do not upgrade/downgrade the NPU stack.')
        suffix = '' if expected is not None else ' (not pinned)'
        print(f'[core-check] {name}=={actual}{suffix}')

    # Guard against a CUDA/GPU torch sneaking in via an unconstrained dependency.
    cuda_dists = sorted(d.metadata['Name'] for d in distributions()
                        if d.metadata['Name'] and CUDA_DIST_RE.match(canon(d.metadata['Name'])))
    if cuda_dists:
        raise RuntimeError('CUDA/GPU packages found in an NPU image: '
                           + ', '.join(cuda_dists)
                           + '. Pin torch/torch_npu via deploy/constraints-npu.txt.')

    try:
        triton_ascend = version('triton-ascend')
    except PackageNotFoundError:
        raise RuntimeError('triton-ascend is missing; this image requires triton-ascend==3.2.2')
    if triton_ascend != '3.2.2':
        raise RuntimeError(f'This image is based on triton-ascend==3.2.2, found {triton_ascend}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
