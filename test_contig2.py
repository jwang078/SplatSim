#!/usr/bin/env python3
import torch

# Create tensor similar to what we have
t = torch.empty(1, device='cuda').contiguous()[:0].reshape(0, 0)
print(f"Original:")
print(f"  numel={t.numel()}, storage_size={t.untyped_storage().size()}")

# Test what C++ code does: check if numel==0, and if so use nullptr, else call contiguous().data_ptr()
if t.numel() == 0:
    print("Would use nullptr (correct)")
else:
    t_contig = t.contiguous()
    print(f"After contiguous():")
    print(f"  numel={t_contig.numel()}, storage_size={t_contig.untyped_storage().size()}")
    try:
        ptr = t_contig.data_ptr()
        print(f"  data_ptr() = {hex(ptr)}")
    except Exception as e:
        print(f"  data_ptr() failed: {e}")

# But what if we call .contiguous() BEFORE checking numel?
print("\nTesting contiguous() before numel() check:")
t2 = torch.empty(1, device='cuda').contiguous()[:0].reshape(0, 0)
t2_contig = t2.contiguous()
print(f"  after .contiguous(): numel={t2_contig.numel()}, storage_size={t2_contig.untyped_storage().size()}")
if t2_contig.numel() == 0:
    print("  numel is 0, would use nullptr")
try:
    ptr2 = t2_contig.data_ptr()
    print(f"  data_ptr() = {hex(ptr2)} (should be 0x0)")
except Exception as e:
    print(f"  data_ptr() FAILED: {e}")
