#!/usr/bin/env python3
import torch

# Test if slice preserves storage
t = torch.empty(1, device='cuda', dtype=torch.uint8).contiguous().slice(0, 0, 0)
print(f"After .slice(0, 0, 0):")
print(f"  shape={t.shape}, numel={t.numel()}")
print(f"  has_storage={t.untyped_storage().size() > 0}, storage_size={t.untyped_storage().size()}")
try:
    ptr = t.data_ptr()
    print(f"  data_ptr() = {hex(ptr)}")
except Exception as e:
    print(f"  data_ptr() FAILED: {e}")
