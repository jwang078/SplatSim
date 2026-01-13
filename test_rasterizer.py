#!/usr/bin/env python3
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Import the rasterization module
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
print(f"✓ Successfully imported diff_gaussian_rasterization")

# Create some test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 100  # number of Gaussians

means3D = torch.randn(N, 3, device=device)
means2D = torch.zeros_like(means3D)
opacity = torch.rand(N, 1, device=device)
scales = torch.rand(N, 3, device=device)
rotations = torch.randn(N, 4, device=device)
rotations = rotations / rotations.norm(dim=-1, keepdim=True)  # normalize quaternions
shs = torch.randn(N, 16, 3, device=device)

# Create rasterization settings
W, H = 640, 480
tanfovx = tanfovy = 0.5
bg = torch.zeros(3, device=device)
viewmatrix = torch.eye(4, device=device)
projmatrix = torch.eye(4, device=device)
campos = torch.zeros(3, device=device)

raster_settings = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg,
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=3,
    campos=campos,
    prefiltered=False,
    debug=False,
    antialiasing=False
)

rasterizer = GaussianRasterizer(raster_settings=raster_settings)
print(f"✓ Created rasterizer")

try:
    # Test with empty colors_precomp and cov3D_precomp (the problematic case)
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        shs=shs,
        colors_precomp=None,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None
    )
    print(f"✓ Rasterization successful!")
    print(f"  Output shape: {rendered_image.shape}")
    print(f"  Output device: {rendered_image.device}")
    print(f"  Output dtype: {rendered_image.dtype}")
    print(f"\n✓✓✓ ALL TESTS PASSED! The fix works correctly. ✓✓✓")
except Exception as e:
    print(f"✗ Rasterization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
