#!/usr/bin/env python3
"""
Diagnostic test for causal masking correctness.

Runs the causal kernel and compares against PyTorch's SDPA with is_causal=True.
Prints per-position error patterns to identify WHERE the mask is wrong.

Usage:
    python tools/debug/causal_test.py
"""

import torch
import torch.nn.functional as F
import flash_attention
from flash_helpers.kernel_configs import FlashForwardKernelConfig, DType


def causal_reference(q, k, v):
    """Reference causal attention using F.scaled_dot_product_attention.

    q/k/v: (B, S, H, D) — our layout
    Returns: (B, S, H, D)
    """
    # SDPA expects (B, H, S, D)
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=True)
    return out.permute(0, 2, 1, 3)


def run_causal_kernel(cfg, q, k, v):
    """Run our causal flash attention kernel."""
    return flash_attention.forward(cfg, q, k, v)


def analyze_errors(ref, out, seq_len):
    """
    Print per-position error pattern.
    Distinguishes errors in upper-triangle (should be masked / near-zero)
    vs lower-triangle (should be non-trivial).
    """
    diff = (ref - out).abs()  # (B, S, H, D)
    max_diff = diff.max().item()
    print(f"  Max absolute error: {max_diff:.6f}")

    # Collapse batch and head dims, look at seq pattern
    # diff shape: (B, S, H, D)
    B, S, H, D = diff.shape
    diff_seq = diff.max(dim=-1).values  # (B, S, H) — worst d_head error per position
    diff_seq = diff_seq.max(dim=-1).values  # (B, S) — worst head error
    diff_seq = diff_seq.max(dim=0).values  # (S,) — worst batch error

    print(f"  Max error per query position (seq_len={S}):")
    for i in range(0, S, 8):
        vals = diff_seq[i:i+8].tolist()
        print(f"    q[{i:3d}-{i+7:3d}]: " + " ".join(f"{v:.4f}" for v in vals))

    # For a single (batch=0, head=0), show the attention matrix errors
    # by comparing what each query position sees
    q0h0_ref = ref[0, :, 0, :]   # (S, D)
    q0h0_out = out[0, :, 0, :]   # (S, D)
    q0h0_diff = (q0h0_ref - q0h0_out).abs()  # (S, D)

    # Max error per query row
    row_errs = q0h0_diff.max(dim=-1).values  # (S,)
    print(f"\n  Per-row max error (batch=0, head=0):")
    upper_errs = []  # rows with mostly upper-triangle issues (early rows attend fewer keys)
    for i in range(min(S, 32)):
        print(f"    row {i:3d}: {row_errs[i].item():.6f}")


def main():
    device = "cuda:0"
    dtype = torch.float16

    # N_HEADS=16 is hardcoded in GMemStride, so H must be 16.
    # seq_len=128 gives exactly 2 Q/KV blocks of size 64 — small enough to debug.
    B, S, H, D = 1, 128, 16, 128
    print(f"Test shape: (B={B}, S={S}, H={H}, D={D})")
    print()

    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, dtype=dtype, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    ref = causal_reference(q, k, v)

    # Test no-swizzle causal kernel
    cfg_no_swizzle = FlashForwardKernelConfig(
        dtype=DType.FP16,
        d_head=D,
        B_r=64,
        B_c=64,
        n_warps=4,
        async_copy=True,
        eager_load_blocks=False,
        swizzled=False,
        Q_mma_load_K_tiles=0,
        K_mma_load_K_tiles=0,
        V_mma_load_K_tiles=0,
        mma_double_buffer_loads=False,
        optimized_softmax=False,
        causal=True,
    )

    cfg_swizzle = FlashForwardKernelConfig(
        dtype=DType.FP16,
        d_head=D,
        B_r=64,
        B_c=64,
        n_warps=4,
        async_copy=True,
        eager_load_blocks=False,
        swizzled=True,
        Q_mma_load_K_tiles=0,
        K_mma_load_K_tiles=0,
        V_mma_load_K_tiles=0,
        mma_double_buffer_loads=False,
        optimized_softmax=False,
        causal=True,
    )

    # Non-causal reference for checking if mask is applied at all
    ref_noncausal = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=False
    ).permute(0, 2, 1, 3)

    for name, cfg in [("no-swizzle causal", cfg_no_swizzle), ("swizzle causal", cfg_swizzle)]:
        print(f"--- {name} ---")
        try:
            out = run_causal_kernel(cfg, q, k, v)
            analyze_errors(ref, out, S)
            # Check if mask is being applied at all
            diff_vs_noncausal = (out - ref_noncausal).abs().max().item()
            print(f"  Max diff vs non-causal output: {diff_vs_noncausal:.6f}")
            if diff_vs_noncausal < 1e-4:
                print("  WARNING: causal output looks identical to non-causal — mask may not be applied!")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    # Also test non-causal versions to make sure we didn't break anything
    cfg_noncausal = FlashForwardKernelConfig(
        dtype=DType.FP16,
        d_head=D,
        B_r=64,
        B_c=64,
        n_warps=4,
        async_copy=True,
        eager_load_blocks=False,
        swizzled=False,
        Q_mma_load_K_tiles=0,
        K_mma_load_K_tiles=0,
        V_mma_load_K_tiles=0,
        mma_double_buffer_loads=False,
        optimized_softmax=False,
        causal=False,
    )
    ref_noncausal = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=False
    ).permute(0, 2, 1, 3)

    print("--- non-causal baseline (should pass) ---")
    try:
        out_noncausal = run_causal_kernel(cfg_noncausal, q, k, v)
        diff = (ref_noncausal - out_noncausal).abs().max().item()
        print(f"  Max error vs SDPA non-causal: {diff:.6f}")
    except Exception as e:
        print(f"  ERROR: {e}")


def gqa_reference(q, k, v):
    """Reference GQA attention using F.scaled_dot_product_attention.

    q: (B, S, H, D)   — full Q heads
    k: (B, S, G, D)   — fewer KV heads (G = H / groups)
    v: (B, S, G, D)
    Returns: (B, S, H, D)
    """
    B, S, H, D = q.shape
    G = k.size(2)
    groups = H // G
    # Expand K/V from (B,S,G,D) to (B,S,H,D) by repeating each KV head groups times
    k_exp = k.repeat_interleave(groups, dim=2)  # (B, S, H, D)
    v_exp = v.repeat_interleave(groups, dim=2)
    # SDPA expects (B, H, S, D)
    out = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k_exp.permute(0, 2, 1, 3),
        v_exp.permute(0, 2, 1, 3),
    )
    return out.permute(0, 2, 1, 3)


def test_gqa():
    device = "cuda:0"
    dtype = torch.float16
    B, S, H, D = 1, 128, 16, 128

    print(f"\n{'='*60}")
    print(f"GQA Test: (B={B}, S={S}, H={H}, D={D})")

    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, dtype=dtype, device=device)

    for n_kv_heads, label in [(8, "G=2"), (4, "G=4")]:
        k = torch.randn(B, S, n_kv_heads, D, dtype=dtype, device=device)
        v = torch.randn_like(k)
        ref = gqa_reference(q, k, v)

        for swizzled, sw_label in [(False, "no-swizzle"), (True, "swizzle")]:
            cfg = FlashForwardKernelConfig(
                dtype=DType.FP16,
                d_head=D,
                B_r=64,
                B_c=64,
                n_warps=4,
                async_copy=True,
                eager_load_blocks=False,
                swizzled=swizzled,
                Q_mma_load_K_tiles=0,
                K_mma_load_K_tiles=0,
                V_mma_load_K_tiles=0,
                mma_double_buffer_loads=False,
                optimized_softmax=False,
                causal=False,
                n_kv_heads=n_kv_heads,
            )
            name = f"{label} {sw_label}"
            print(f"\n--- {name} ---")
            try:
                out = flash_attention.forward(cfg, q, k, v)
                diff = (ref - out).abs().max().item()
                print(f"  Max absolute error vs GQA reference: {diff:.6f}")
                if diff < 5e-3:
                    print("  PASSED")
                else:
                    print("  FAILED — errors too large")
            except Exception as e:
                print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
    test_gqa()
