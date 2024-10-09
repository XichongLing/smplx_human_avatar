#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr


import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import torch.nn.functional as F
import os
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

import matplotlib.pyplot as plt
import trimesh
import igl

class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, cfg):
        self.cfg = cfg

        # two modes: SH coefficient or feature
        self.use_sh = cfg.use_sh
        self.active_sh_degree = 0
        if self.use_sh:
            self.max_sh_degree = cfg.sh_degree
            self.feature_dim = (self.max_sh_degree + 1) ** 2
        else:
            self.feature_dim = cfg.feature_dim

        self._xyz = torch.empty(0)
        # the joint weights of xyz, size of (N, J) 
        self._xyz_J = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._label = torch.empty(0)
        self.setup_functions()

    def clone(self):
        cloned = GaussianModel(self.cfg)

        properties = ["active_sh_degree",
                      "non_rigid_feature",
                      ]
        for property in properties:
            if hasattr(self, property):
                setattr(cloned, property, getattr(self, property))

        parameters = ["_xyz",
                      "_features_dc",
                      "_features_rest",
                      "_scaling",
                      "_rotation",
                      "_opacity",
                      "_label",]
        for parameter in parameters:
            setattr(cloned, parameter, getattr(self, parameter) + 0.)

        return cloned

    def init_fwd_transform(self, transl, root_orient_mat):
        initial_transform = torch.eye(4, device="cuda").unsqueeze(0).repeat(self.get_xyz.shape[0], 1, 1)
        initial_transform[self._label[:, 0] == 1, :3, :] = torch.matmul(root_orient_mat.cuda(), initial_transform[self._label[:, 0] == 1, :3, :])
        initial_transform[self._label[:, 0] == 1, :3, 3] += torch.tensor(transl).reshape(1, 3).cuda()  # add global offset
        self.fwd_transform = initial_transform

    def set_fwd_transform(self, T_fwd):
        self.fwd_transform = T_fwd

    def get_fwd_transform(self):
        return self.fwd_transform

    def get_fwd_transform_by_category(self, label):
        return self.fwd_transform[self._label[:, 0] == label]
    
    def set_fwd_transform_by_category(self, label, T_fwd):
        self.fwd_transform[self._label[:, 0] == label] = T_fwd


    def set_skinning_weights(self, skinning_weights):
        self.skinning_weights = skinning_weights

    def color_by_opacity(self):
        cloned = self.clone()
        cloned._features_dc = self.get_opacity.unsqueeze(-1).expand(-1,-1,3)
        cloned._features_rest = torch.zeros_like(cloned._features_rest)
        return cloned

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._label
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._label,
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    def get_xyz_by_category(self, label):
        return self._xyz[self._label[:, 0] == label]

    def set_xyz_by_category(self, label, xyz):
        self._xyz[self._label[:, 0] == label] = xyz

    def get_rotation_by_category(self, label):
        return self._rotation[self._label[:, 0] == label]
    
    def get_scaling_by_category(self, label):
        return self._scaling[self._label[:, 0] == label]
    
    def set_rotation_by_category(self, label, rotation):
        self._rotation[self._label[:, 0] == label] = rotation
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    @property
    
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_xyz_J(self):
        return self._xyz_J
    @property
    def get_label(self):
        return self._label

    def set_xyz_J(self, xyz_J):
        self._xyz_J = xyz_J

    def get_covariance(self, scaling_modifier = 1):
        if hasattr(self, 'rotation_precomp'):
            return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation_precomp)
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if not self.use_sh:
            return
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_opacity_loss(self):
        # opacity classification loss
        opacity = self.get_opacity
        eps = 1e-6
        loss_opacity_cls = -(opacity * torch.log(opacity + eps) + (1 - opacity) * torch.log(1 - opacity + eps)).mean()
        return {'opacity': loss_opacity_cls}

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        if self.use_sh:
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((fused_color.shape[0], 1, self.feature_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # Mean of the squared distance to the knn points
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)  # Isotropic scaling
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # Map from 0.1 back to (-inf, inf) using inverse sigmoid

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._label = torch.zeros(self.get_xyz.shape[0], device="cuda").view(-1, 1)

    def create_from_multi_pcd(self, pcd_list, spatial_lr_scale=1.):
        self.spatial_lr_scale = spatial_lr_scale
        
        category = 0
        label = torch.empty(0).cuda()
        fused_point_clouds = torch.empty(0).cuda()
        fused_colors = torch.empty(0).cuda()
        for pcd in pcd_list:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            print("number of points in category ", category, " : ", fused_point_cloud.shape[0])
            fused_point_clouds = torch.cat((fused_point_clouds, fused_point_cloud), dim=0).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            fused_colors = torch.cat((fused_colors, fused_color), dim=0).float().cuda()
            label = torch.cat((label, torch.ones(fused_point_cloud.shape[0], device="cuda") * category), dim=0)
            category += 1
        self._label = label.view(-1, 1)
        if self.use_sh:
            features = torch.zeros((fused_colors.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_colors
            features[:, 3:, 1:] = 0.0
        else:
            features = torch.zeros((fused_colors.shape[0], 1, self.feature_dim)).float().cuda()


        dist2s = torch.empty(0).cuda()
        for pcd in pcd_list:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            dist2s = torch.cat((dist2s, dist2), dim=0).float().cuda()
        # Mean of the squared distance to the knn points
        scales = torch.log(torch.sqrt(dist2s))[...,None].repeat(1, 3)  # Isotropic scaling
        rots = torch.zeros((fused_point_clouds.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_clouds.shape[0], 1), dtype=torch.float, device="cuda"))
        # Map from 0.1 back to (-inf, inf) using inverse sigmoid

        self._xyz = nn.Parameter(fused_point_clouds.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        print("dimension of xyz : ", self._xyz.shape)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        feature_ratio = 20.0 if self.use_sh else 1.0
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / feature_ratio, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_weights(self, path):
        joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
        joint_colors = joint_colors[:,:3]
        points = self.get_xyz.detach().cpu()
        pred = self.get_xyz_J.detach().cpu()
        maxjoint_pred_idx = torch.argmax(pred, dim=1)
        # maxjoint_pred_idx = torch.topk(pred, k=2, dim=1)[1][..., 1]
        # import ipdb; ipdb.set_trace()
        pred_colors = joint_colors[maxjoint_pred_idx]
        with open(path, 'w') as f:
            for point, color in zip(points, pred_colors):
                # Write XYZRGB data to the file
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        print("XYZRGB file created successfully.")


    def save_aabb_deformed_ply(self, path, aabb):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # xyz = aabb.normalize(self._xyz, sym=True).detach().cpu().numpy()

        # normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()

        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # el = PlyElement.describe(elements, 'vertex')
        # PlyData([el]).write(path)
        joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
        joint_colors = joint_colors[:,:3]
        points = aabb.normalize(self._xyz, sym=True).detach().cpu().numpy()
        pred = self.get_xyz_J.detach().cpu()
        maxjoint_pred_idx = torch.argmax(pred, dim=1)
        pred_colors = joint_colors[maxjoint_pred_idx]
        with open(path, 'w') as f:
            for point, color in zip(points, pred_colors):
                # Write XYZRGB data to the file
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        print("XYZRGB file created successfully.")

        
    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._label = self._label[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        # convert new output of densification requiring indices to optimizable tensors
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def extract_hand_points(self):
        # hands joints index
        hand_J = list(range(25, 55)) + [20, 21]
        # may not be the same as get_xyz_J since it has already been cloned
        n_init_points = self.get_xyz.shape[0]
        hand_points_mask = torch.zeros(n_init_points, dtype=bool, device="cuda")
        
        if self.get_xyz_J.shape[0] != 1:
            xyz_argmaxJ = torch.argmax(self.get_xyz_J, dim=-1)
            # hand_points_mask[:xyz_argmaxJ.shape[0]] = torch.where(xyz_argmaxJ == 40 , True, False)
            for h in hand_J:
                hand_points_mask[:xyz_argmaxJ.shape[0]] = torch.logical_or( hand_points_mask[:xyz_argmaxJ.shape[0]], torch.where(xyz_argmaxJ == h , True, False))
            # hand_points_mask[:xyz_argmaxJ.shape[0]] = torch.logical_or( *[ xyz_argmaxJ == h for h in hand_J ] )

        return hand_points_mask
    
    # when scale is high, after clone, that's why it has 0-padded grad
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, hand_extra_density=False):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # why now without norms
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        
        if hand_extra_density:
            # hand points have lower threshold
            hand_pts_mask = self.extract_hand_points()
            hand_pts_mask = torch.logical_and( hand_pts_mask,
                                               torch.where(padded_grad>= grad_threshold /4, True, False))
            selected_pts_mask = torch.logical_or(selected_pts_mask, hand_pts_mask)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # create N new points for each selected point
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        new_label = self._label[selected_pts_mask].repeat(N, 1)
        self._label = torch.cat((self._label, new_label), dim=0)
        # delete the original points, now the mask size is original mask + N times the selected points 
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    # when scale is low
    def densify_and_clone(self, grads, grad_threshold, scene_extent, hand_extra_density=False):
        # Extract points that satisfy the gradient condition
        # gradient norm along last dim should be greater than grad_threshold
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        if hand_extra_density:
        # hand points have lower threshold
            hand_pts_mask = self.extract_hand_points()
            hand_pts_mask = torch.logical_and( hand_pts_mask,
                                               torch.where(torch.norm(grads, dim=-1)>= grad_threshold /4, True, False))        

            selected_pts_mask = torch.logical_or(selected_pts_mask,hand_pts_mask)
            
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # import ipdb; ipdb.set_trace()
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        self._label = torch.cat((self._label, self._label[selected_pts_mask]), dim=0)


    def densify_and_prune(self, opt, scene, max_screen_size):
        extent = scene.cameras_extent

        max_grad = opt.densify_grad_threshold
        min_opacity = opt.opacity_threshold

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def get_segmentation(self,):
        # to test, set the body to blue and the garments to red
        segmentation = torch.zeros((self._label.shape[0], 3), device="cuda")
        segmentation[self._label[:, 0] == 1] = torch.tensor([1., 0, 0], device="cuda") 
        segmentation[self._label[:, 0] == 0] = torch.tensor([0, 0, 1.], device="cuda")  
        return segmentation, self._label
    
    def extract_virtual_bones(self,):
        num_vb = 80
        garm_xyz = self.get_xyz_by_category(1)
        mask = torch.rand(garm_xyz.shape[0]).argsort(0) < num_vb
        return garm_xyz[mask].detach()
        # random_mask = 
        # return self._xyz[self._label[:, 0] == 1].