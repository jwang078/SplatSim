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

    Ks = splatsim_camera.intrinsic_matrix.unsqueeze(0)  # [1, 3, 3]

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

    return {
        "render": rendered_image,
        "depth": torch.zeros(1, H, W, device="cuda"),  # placeholder
    }
