#!/usr/bin/env python3
import torch

# Create a tensor with storage but zero numel
t = torch.empty(1, device='cuda').contiguous()[:0].reshape(0, 0)
print(f"Original tensor:")
print(f"  shape={t.shape}, numel={t.numel()}, has_storage={t.untyped_storage().size() > 0}, storage_size={t.untyped_storage().size()}")

# Call contiguous on it
t_contig = t.contiguous()
print(f"\nAfter .contiguous():")
print(f"  shape={t_contig.shape}, numel={t_contig.numel()}, has_storage={t_contig.untyped_storage().size() > 0}, storage_size={t_contig.untyped_storage().size()}")

# Try to get data pointer
try:
    ptr = t_contig.data_ptr()
    print(f"  data_ptr() succeeded: {hex(ptr)}")
except Exception as e:
    print(f"  data_ptr() failed: {e}")
