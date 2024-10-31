#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import depth_to_normal

def render(data,
           data_t,
           iteration,
           scene,
           pipe,
           bg_color : torch.Tensor,
           kernel_size = 0,
           scaling_modifier = 1.0,
           rasterizer_type = '3DGS',
           override_color = None,
           compute_loss=True,
           return_opacity=False, 
           return_segmentation=False,
           return_masked_rendering=False, 
           require_coord : bool = True, require_depth : bool = True,
           return_normal=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    pc, loss_reg, colors_precomp, colors_segmentation, gaussian_labels, joint_colors, non_rigid_xyz = scene.convert_gaussians(data, data_t, iteration, compute_loss)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    if rasterizer_type == 'VCRGS':
        screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points_densify.retain_grad()
        except:
            pass   

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    if rasterizer_type == 'RaDe':
        raster_settings = GaussianRasterizationSettings(
            image_height=int(data.image_height),
            image_width=int(data.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            kernel_size = kernel_size,
            scale_modifier=scaling_modifier,
            viewmatrix=data.world_view_transform,
            projmatrix=data.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=data.camera_center,
            prefiltered=False,
            require_coord=require_coord,
            require_depth=require_depth,
            debug=pipe.debug
        )
    elif rasterizer_type == '3DGS':
        raster_settings = GaussianRasterizationSettings(
            image_height=int(data.image_height),
            image_width=int(data.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=data.world_view_transform,
            projmatrix=data.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=data.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
    elif rasterizer_type == '2DGS':
        raster_settings = GaussianRasterizationSettings(
            image_height=int(data.image_height),
            image_width=int(data.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=data.world_view_transform,
            projmatrix=data.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=data.camera_center,
            prefiltered=False,
            debug=False,
            # pipe.debug
        )
    elif rasterizer_type == 'VCRGS':
        raster_settings = GaussianRasterizationSettings(
            image_height=int(data.image_height),
            image_width=int(data.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=data.world_view_transform,
            projmatrix=data.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=data.camera_center,
            prefiltered=False,
            debug=pipe.debug,
            f_count=0,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    if rasterizer_type == 'VCRGS':
        means2D_densify = screenspace_points_densify

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    # import ipdb; ipdb.set_trace()
    if pipe.compute_cov3D_python:
        if rasterizer_type == '2DGS':
            # currently don't support normal consistency loss if use precomputed covariance
            splat2world = pc.get_covariance(scaling_modifier)
            W, H = data.image_width, data.image_height
            near, far = data.znear, data.zfar
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, far-near, near],
                [0, 0, 0, 1]]).float().cuda().T
            world2pix =  data.full_proj_transform @ ndc2pix
            cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    normals_precomp = None
    is_all = True # not sure what this parameter does
    if return_normal:
        normal = pc.get_normal(is_all=is_all)
        # convert normal direction to the camera; calculate the normal in the camera coordinate
        view_dir = means3D - data.camera_center
        normal   = normal * ((((view_dir * normal).sum(dim=-1) > 0) * 1 - 0.5) * 2)[..., None]
        R_w2c = torch.tensor(data.R.T).cuda().to(torch.float32)
        normals_precomp = normal @ R_w2c.transpose(0, 1)        # camera coordinate
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # pc_joint_labels = torch.argmax(pc.skinning_weights, dim=1)
    # pc_left_hand_mask = torch.logical_or(pc_joint_labels == 20, torch.logical_and(pc_joint_labels >= 25, pc_joint_labels <= 39))
    # pc_right_hand_mask = torch.logical_or(pc_joint_labels == 21, torch.logical_and(pc_joint_labels >= 40, pc_joint_labels <= 54))

    # left_hand_mask, _ = rasterizer(
    #     means3D=means3D,
    #     means2D=means2D,
    #     shs=None,
    #     colors_precomp=pc_left_hand_mask.float().unsqueeze(1).repeat(1, 3),
    #     opacities=torch.ones_like(opacity).to(opacity.device),
    #     scales=scales,
    #     rotations=rotations,
    #     cov3D_precomp=cov3D_precomp)

    # right_hand_mask, _ = rasterizer(
    #     means3D=means3D,
    #     means2D=means2D,
    #     shs=None,
    #     colors_precomp=pc_right_hand_mask.float().unsqueeze(1).repeat(1, 3),
    #     opacities=torch.ones_like(opacity).to(opacity.device),
    #     scales=scales,
    #     rotations=rotations,
    #     cov3D_precomp=cov3D_precomp
    # )

    # import ipdb; ipdb.set_trace()

    if rasterizer_type in ['RaDe', '3DGS', '2DGS']:
        render_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        
    elif rasterizer_type in ['VCRGS']:
        render_output = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_densify = means2D_densify,
            shs = shs,
            colors_precomp = colors_precomp,
            normals_precomp = normals_precomp,
            semantics_precomp = None,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            dirs = None,
            inside = None)

    if rasterizer_type == 'RaDe':
        rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = render_output
    elif rasterizer_type == '3DGS':
        rendered_image, radii = render_output
    elif rasterizer_type == '2DGS':
        rendered_image, radii, allmap = render_output
            # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (data.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1; 
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
        
        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = depth_to_normal(data, surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * (render_alpha).detach()

    elif rasterizer_type == 'VCRGS':
        rendered_out, radii = render_output
        chs = [3, 1, 3, 1]
        rendered_image, rendered_depth, rendered_normal, rendered_alpha = rendered_out[:sum(chs)].split(chs, dim=0)


    opacity_image = None

    if return_opacity:
        if rasterizer_type in ['3DGS','RaDe', '2DGS']:
            render_output = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device), # Here colors are used as opacities
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)
        elif rasterizer_type == 'VCRGS':
            render_output = rasterizer(
                means3D = means3D,
                means2D = means2D,
                means2D_densify = means2D_densify,
                shs = shs,
                colors_precomp = torch.ones(opacity.shape[0], 3, device=opacity.device), # Here colors are used as opacities,
                normals_precomp = normals_precomp,
                semantics_precomp = None,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                dirs = None,
                inside = None)

        if rasterizer_type in ['RaDe']:
            opacity_image, _, _, _, _, _, _, _ = render_output
        elif rasterizer_type in ['3DGS']:
            opacity_image,_ = render_output
        elif rasterizer_type in ['2DGS']:
            opacity_image, _, _ = render_output
        elif rasterizer_type in ['VCRGS']:
            rendered_out, radii = render_output
            chs = [3, 1, 3, 1]
            opacity_image, _, _, _ = rendered_out[:sum(chs)].split(chs, dim=0)

        opacity_image = opacity_image[:1]  # Only the first channel is needed

    segmentation_image = None
    if return_segmentation:
        colors_segmentation = colors_segmentation.to(means3D.device)
        if rasterizer_type in ['3DGS','RaDe', '2DGS']:
            render_output = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=colors_segmentation,
                opacities=torch.ones(opacity.shape[0], device=opacity.device),
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)
        elif rasterizer_type in ['VCRGS']:    
            render_output = rasterizer(
                means3D = means3D,
                means2D = means2D,
                means2D_densify = means2D_densify,
                shs = shs,
                colors_precomp = colors_segmentation,
                normals_precomp = normals_precomp,
                semantics_precomp = None,
                opacities = torch.ones(opacity.shape[0], device=opacity.device),
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                dirs = None,
                inside = None)
        if rasterizer_type == 'RaDe':
            segmentation_image, _, _, _, _, _, _, _ = render_output
        elif rasterizer_type == '3DGS':
            segmentation_image, _ = render_output
        elif rasterizer_type == '2DGS':
            segmentation_image, _, _ = render_output
        elif rasterizer_type == 'VCRGS':
            rendered_out, radii = render_output
            chs = [3, 1, 3, 1]
            segmentation_image, _, _, _ = rendered_out[:sum(chs)].split(chs, dim=0)
        
    masked_rendering = None
    if return_masked_rendering:
        means3D_masked = means3D[gaussian_labels == 1]
        means2D_masked = means2D[gaussian_labels == 1]
        opacity_masked = opacity[gaussian_labels == 1]
        colors_masked = colors_precomp[gaussian_labels == 1]
        scales_masked = scales[gaussian_labels == 1]
        rotations_masked = rotations[gaussian_labels == 1]
        cov3D_masked = cov3D_precomp[gaussian_labels == 1]

        render_output = rasterizer(
            means3D=means3D_masked,
            means2D=means2D_masked,
            shs=None,
            colors_precomp=colors_masked,
            opacities=torch.ones(opacity_masked.shape[0], device=opacity_masked.device),
            scales=scales_masked,
            rotations=rotations_masked,
            cov3D_precomp=cov3D_masked)
        if rasterizer_type == 'RaDe':
            masked_rendering, _, _, _, _, _, _, _ = render_output
        elif rasterizer_type == '3DGS':
            masked_rendering, _ = render_output
        elif rasterizer_type == '2DGS':
            masked_rendering, _, _ = render_output
        elif rasterizer_type == 'VCRGS':
            rendered_out, _ = render_output
            chs = [3, 1, 3, 1]
            masked_rendering, _, _, _ = rendered_out[:sum(chs)].split(chs, dim=0)


    joint_image = None
    if pipe.visualize_joints:
        joint_colors = joint_colors.to(means3D.device)
        non_rigid_xyz = non_rigid_xyz.to(means3D.device)
        if rasterizer_type in ['3DGS','RaDe', '2DGS']:
            render_output = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=None,
                colors_precomp=joint_colors,
                opacities=torch.ones(opacity.shape[0], device=opacity.device),
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)
            joint_image, _ = render_output
            render_output = rasterizer(
                means3D=non_rigid_xyz,
                means2D=means2D,
                shs=None,
                colors_precomp=joint_colors,
                opacities=torch.ones(opacity.shape[0], device=opacity.device),
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)
            non_rigid_joint_image, _ = render_output




    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if rasterizer_type == 'RaDe':

        return {"deformed_gaussian": pc, # Deformed Gaussians object
                "render": rendered_image, # Rendered image: (3, H, W)
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0, # If radius is 0, the Gaussian was not visible.
                "radii": radii,
                "loss_reg": loss_reg,
                "opacity_render": opacity_image,
                "segmentation_render": segmentation_image,
                "masked_rendering": masked_rendering,
                # "left_hand_mask": left_hand_mask,
                # "right_hand_mask": right_hand_mask,
                "expected_depth": rendered_expected_depth,
                "median_depth": rendered_median_depth,
                "normal": rendered_normal,
                }

    elif rasterizer_type == '3DGS':
        return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": opacity_image,
            "segmentation_render": segmentation_image,
            "masked_rendering": masked_rendering,
            "joint_image": joint_image,
            "non_rigid_joint_image": non_rigid_joint_image  
        }
    
    elif rasterizer_type == '2DGS':
        return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": opacity_image,
            "segmentation_render": segmentation_image,
            "masked_rendering": masked_rendering,
            "depth_median": render_depth_median,    
            "expected_depth": render_depth_expected,
            "normal": render_normal,
            "distortion": render_dist,
            "depth_median": render_depth_median,
            "surf_normal": surf_normal,
            "surf_depth": surf_depth,
        }
    elif rasterizer_type == 'VCRGS':
        return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": opacity_image,
            "segmentation_render": segmentation_image,
            "masked_rendering": masked_rendering,
            "expected_depth": rendered_depth,
            "normal": rendered_normal,
            "alpha": rendered_alpha,
        }
