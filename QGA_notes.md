## GQA Implementation Notes

In GQA with $G$ groups, query head $h$ uses KV head $\left\lfloor \frac{h}{G} \right\rfloor$. The two main changes in global memory (gmem) access are:

1. **KV head selection:**  
   $kv\_head = \left\lfloor \frac{head}{gqa\_groups} \right\rfloor$ instead of $head$
2. **Row stride for K/V:**  
   Sequence stride for K/V is $n\_kv\_heads \times d\_head$ (not $N\_HEADS \times d\_head$)

Everything inside shared memory (the tiled computation, softmax, PV matmul) is identical to MHA.

---

### File-by-File Changes

| File                        | What Changed                                                                                      |
|-----------------------------|---------------------------------------------------------------------------------------------------|
| `flash_attention.cuh`       | `kv_batch_stride` added to `ForwardKernelArgs`; `n_kv_heads=16` (15th field, default = MHA) added to config and `operator<` |
| `static_kernel_configuration.cuh` | `n_kv_heads`, `gqa_groups` as `constexpr`s; `GMemStrideKV = n_kv_heads × d_head` replaces `GMemStride` for K/V so tile copies use the right row stride |
| `forward_kernel.cuh`        | Splits into `QO_sample_head_offset` (uses `head`) and `KV_sample_head_offset` (uses `head / gqa_groups` and `kv_batch_stride`) |
| `flash_kernels.cuh`         | 4 new entries: G=2 (8 KV heads) ×2 swizzle variants, G=4 (4 KV heads) ×2 swizzle variants         |
| `flash_attention.cu`        | Passes `n_kv_heads`; uses `TK.stride(0)` as `kv_batch_stride`; relaxed shape check allows K/V to have fewer heads than Q |
| `kernel_configs.py`         | `n_kv_heads: int = 16` field; `gqa{n}h` in short_form; 15-param parser support                    |

---

### Backward Compatibility

- All existing MHA kernels get `n_kv_heads=16` by default → `gqa_groups=1` → `kv_head = head/1 = head`, `kv_batch_stride = TQ.stride(0)` for MHA tensors.
- Zero change in behavior. ✓

---

### Test Command on GPU

```bash
python tools/debug/causal_test.py   # runs GQA test at the bottom too
```

---

Let me know if you want further formatting or details!