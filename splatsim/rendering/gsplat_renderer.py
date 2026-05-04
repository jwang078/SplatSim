import math

import torch
from gsplat.rendering import rasterization


def render_gsplat(splatsim_camera, pc):
    """Render gaussians using gsplat with fisheye or pinhole camera model.

    Args:
        splatsim_camera: SplatSimCamera with camera_model, intrinsic_matrix, radial_coeffs
        pc: GaussianModel instance (same as used by the original renderer)

    Returns:
        dict with "render" key containing [3, H, W] clamped image tensor on CUDA
    """
    camera = splatsim_camera.camera
    W = camera.image_width
    H = camera.image_height

    # Undo column-major transpose (cameras.py stores .transpose(0,1) for OpenGL)
    viewmats = camera.world_view_transform.T.unsqueeze(0)  # [1, 4, 4]

    if splatsim_camera.intrinsic_matrix is not None:
        Ks = splatsim_camera.intrinsic_matrix.unsqueeze(0)  # [1, 3, 3]
    else:
        # Derive pinhole K from FoV when not explicitly provided.
        fx = (W / 2.0) / math.tan(camera.FoVx * 0.5)
        fy = (H / 2.0) / math.tan(camera.FoVy * 0.5)
        K = torch.tensor(
            [[fx, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]],
            device="cuda",
            dtype=torch.float32,
        )
        Ks = K.unsqueeze(0)  # [1, 3, 3]

    radial_coeffs = None
    if splatsim_camera.radial_coeffs is not None:
        radial_coeffs = splatsim_camera.radial_coeffs.unsqueeze(0)  # [1, 4]

    backgrounds = splatsim_camera.background.unsqueeze(0)  # [1, 3]

    render_colors, render_alphas, meta = rasterization(
        means=pc.get_xyz,
        quats=pc.get_rotation,
        scales=pc.get_scaling,
        opacities=pc.get_opacity.squeeze(-1),
        colors=pc.get_features,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=pc.active_sh_degree,
        camera_model=splatsim_camera.camera_model,
        backgrounds=backgrounds,
        radial_coeffs=radial_coeffs,
        with_ut=radial_coeffs is not None,
        packed=False,
        rasterize_mode="antialiased",
        radius_clip=3.0,  # clip gaussians with projected 2D radius > this many tiles
    )

    # gsplat output: [1, H, W, 3] → [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1).clamp(0, 1)

    # For fisheye cameras, gsplat rasterizes over the full W x H rectangle but
    # real lenses only illuminate a circular image region — mask the outside to
    # black to simulate the lens vignette. The image circle may extend past the
    # far edges of the sensor (W - cx, H - cy), so we only bound the radius by
    # the near edges (cx, cy).
    if splatsim_camera.camera_model == "fisheye":
        cx = Ks[0, 0, 2].item()
        cy = Ks[0, 1, 2].item()
        mask_radius = min(cx, cy, W - cx, H - cy) * 1.07
        ys, xs = torch.meshgrid(
            torch.arange(H, device="cuda", dtype=torch.float32),
            torch.arange(W, device="cuda", dtype=torch.float32),
            indexing="ij",
        )
        inside = ((xs - cx) ** 2 + (ys - cy) ** 2) <= mask_radius ** 2
        rendered_image = rendered_image * inside.unsqueeze(0)

    return {
        "render": rendered_image,
        "depth": torch.zeros(1, H, W, device="cuda"),  # placeholder
    }
