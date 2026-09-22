#!/usr/bin/env python
"""Best-effort check that the Qwen3.5 GDN custom operator actually executes.

Used by entrypoint.sh. Exit codes:
  0 success, 2 NPU/stack not available (treated as 'skipped'), 1 failure.
"""
import sys


def main() -> int:
    try:
        import torch
        import torch_npu
    except Exception as e:  # pragma: no cover - env dependent
        print(f'[opcheck] torch/torch_npu unavailable: {e}')
        return 2

    try:
        if not torch_npu.npu.is_available() or torch_npu.npu.device_count() < 1:
            print('[opcheck] no NPU device visible; skipped')
            return 2
        torch.npu.set_device(0)
    except Exception as e:  # pragma: no cover - env dependent
        print(f'[opcheck] NPU init failed: {e}')
        return 2

    try:
        from vllm_ascend.utils import enable_custom_op
        if not enable_custom_op():
            print('[opcheck] vllm-ascend custom op not enabled')
            return 1
    except Exception as e:
        print(f'[opcheck] vllm_ascend unavailable: {e}')
        return 2

    try:
        a_log = torch.zeros(16, device='npu', dtype=torch.float32)
        dt_bias = torch.zeros_like(a_log)
        a = torch.zeros((1, 16), device='npu', dtype=torch.bfloat16)
        b = torch.zeros_like(a)
        g, beta = torch.ops._C_ascend.npu_fused_gdn_gating(a_log, a, b, dt_bias)
        torch.npu.synchronize()
        print(f'[opcheck] npu_fused_gdn_gating = True ({tuple(g.shape)}, {tuple(beta.shape)})')
        return 0
    except Exception as e:
        print(f'[opcheck] npu_fused_gdn_gating = False: {e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
