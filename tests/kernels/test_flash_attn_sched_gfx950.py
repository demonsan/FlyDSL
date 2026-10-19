#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Schedule-IR-emitted gfx950 dual-wave flash attention vs SDPA and the hand-written kernel.

The emitted kernel walks the same op sequence as ``flash_attn_gfx950.py`` (see
``tests/unit/test_sched_emit.py``), so on top of the numerical check against
SDPA its output is expected to be bit-identical to the hand-written kernel.
"""

import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402

if not str(get_rocm_arch()).startswith("gfx950"):
    pytest.skip(f"requires gfx950, got {get_rocm_arch()}", allow_module_level=True)

from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module  # noqa: E402
from kernels.attention.flash_attn_sched_gfx950 import build_flash_attn_sched_gfx950_module  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _sdpa(q, k, v, causal):
    qf, kf, vf = (t.float().permute(0, 2, 1, 3) for t in (q, k, v))
    s = torch.einsum("bhqd,bhkd->bhqk", qf, kf) / math.sqrt(q.shape[-1])
    if causal:
        sq, skv = s.shape[-2], s.shape[-1]
        mask = torch.arange(skv, device=q.device)[None, :] > (torch.arange(sq, device=q.device)[:, None] + skv - sq)
        s = s.masked_fill(mask, float("-inf"))
    return torch.einsum("bhqk,bhkd->bhqd", s.softmax(-1), vf).permute(0, 2, 1, 3)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("B,S,H,D", [(1, 384, 8, 128), (2, 1024, 8, 128), (1, 2048, 16, 64), (2, 700, 4, 128)])
def test_sched_kernel_matches_sdpa_and_hand_kernel(B, S, H, D, dtype, causal):
    torch.manual_seed(B * 7 + S + H + D)
    q, k, v = (torch.randn(B, S, H, D, device="cuda", dtype=dtype) for _ in range(3))
    dt = "bf16" if dtype == torch.bfloat16 else "f16"

    sched_kernel = build_flash_attn_sched_gfx950_module(H, D, causal=causal, dtype_str=dt)
    assert sched_kernel.verify_report.ok
    out = torch.empty_like(q)
    sched_kernel(q, k, v, out, B, S)

    hand_kernel = build_flash_attn_dualwave_swp_module(H, D, causal=causal, dtype_str=dt)
    ref_hand = torch.empty_like(q)
    hand_kernel(q, k, v, ref_hand, B, S)
    torch.cuda.synchronize()

    ref = _sdpa(q, k, v, causal).to(dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    assert torch.equal(out, ref_hand), "schedule-emitted kernel diverges from the hand-written kernel"
