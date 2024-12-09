import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d.ops as ops
import trimesh
import igl

from utils.general_utils import build_rotation, gram_schmidt_batch, to_transform_mat, vert2hetereoply
from models.network_utils import get_skinning_mlp, get_ImplicitNet, get_deformation_mlp, get_bone_encoder, get_hashGrid, HierarchicalPoseEncoder, VanillaCondMLP
from utils.dataset_utils import AABB 
import time
import copy

import matplotlib.pyplot as plt
from scene.gaussian_model import VirtualBone
class Garm_Simulator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera, **kwargs):
        raise NotImplementedError

    def regularization(self):
        return NotImplementedError
    
    def get_xyz_J(self, gaussians):
        return torch.empty(0)


class Identity(Garm_Simulator):
    """ Identity mapping for single frame reconstruction """
    def __init__(self, cfg, metadata, garmsim_args):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera, time_enc, **kwargs):
        tfs = camera.bone_transforms
        # global translation
        trans = tfs[:, :3, 3].mean(0)        
        

        deformed_gaussians = gaussians.clone()
        T_fwd = torch.eye(4, device=gaussians.get_xyz.device).unsqueeze(0).expand(gaussians.get_xyz.shape[0], -1, -1)
        T_fwd = T_fwd.clone()
        T_fwd[:, :3, 3] += + trans.reshape(1, 3)  # add global offset

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]

        # get forward transform by weight * bone_transforms for each Gaussian point
        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        deformed_gaussians.set_fwd_transform(T_fwd.detach())
        return deformed_gaussians, torch.empty(0)

    def regularization(self):
        return {}
    
    def extract_virtual_bones(self, gaussians):
        virtual_bones = gaussians.extract_virtual_bones()
        self.virtual_joints = virtual_bones
        self.num_vb = self.virtual_joints.shape[0]
        return self.virtual_joints

    def get_virtual_joints(self):
        return self.virtual_joints


def create_voxel_grid(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''


# need to change here
class DeformationGraph(Garm_Simulator):
    def __init__(self, cfg, metadata, garm_simulator_args):
        super().__init__(cfg)
        self.vb_mode = garm_simulator_args["vb_mode"]
        self.vb_delay = garm_simulator_args["vb_delay"]

        if self.vb_mode == 'enable':
            self.virtual_bones = metadata["virtual_bones"]
            self.virtual_joints = torch.tensor(self.virtual_bones.vertices).float().cuda()
            self.num_vb = self.virtual_joints.shape[0]
        self.aabb = metadata["aabb"]
        # self.root_orient = metadata["root_orient"]
        # self.trans = metadata["trans"]
        self.update_vb = cfg.update_vb
        self.non_rigid_style = True

        self.trainable_label = garm_simulator_args["trainable_label"]  
        self.vb_model = VirtualBone(metadata['cameras_extent'])
        self.vb_xyz_trainable = garm_simulator_args["vb_xyz_trainable"]   
        self.save_deform = garm_simulator_args["save_deform"]
        self.dir_ply = "assets/garm_debug/{0}".format(garm_simulator_args["dir_save_ply"])


        if self.non_rigid_style:
            self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
            d_cond = self.pose_encoder.n_output_dims
            # add latent code
            self.latent_dim = cfg.get('latent_dim', 0)
            if self.latent_dim > 0:
                d_cond += self.latent_dim
                self.frame_dict = metadata['frame_dict']
                self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)
            d_in = 3
            d_out = 3 + 3 + 3
            self.feature_dim = cfg.get('feature_dim', 0)
            d_out += self.feature_dim

            # output dimension: position + scale + rotation
            self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)

        else:
            self.distill = cfg.distill
            d, h, w = cfg.res // cfg.z_ratio, cfg.res, cfg.res
            self.resolution = (d, h, w)
            if self.distill:
                self.grid = create_voxel_grid(d, h, w).cuda()

            # self.deformation_graph = get_ImplicitNet(cfg)
            # input_dim = 3 + dim(time_enc)
            # output_dim = 9 (rotation) + 3 (translation)
            self.dim_enc_smpl = cfg.bone_encoder.output_dim
            self.dim_enc_time = 9
            # self.deformation_graph = get_deformation_mlp(3, self.dim_enc_smpl + self.dim_enc_time, 12, cfg.deformation_network)
            self.deformation_graph = get_ImplicitNet(cfg.implicitNet)
            self.bone_encoder = get_bone_encoder(cfg.bone_encoder)
            self.hashgrid = get_hashGrid(cfg.hashgrid)

    def forward(self, gaussians, iteration, camera, time_enc, **kwargs):
        return_mat = True
        # Gaussian position
        # # pose encoding
        if self.non_rigid_style:
            rots = camera.rots
            Jtrs = camera.Jtrs
            pose_feat = self.pose_encoder(rots, Jtrs)

            if self.latent_dim > 0:
                frame_idx = camera.frame_id
                if frame_idx not in self.frame_dict:
                    latent_idx = len(self.frame_dict) - 1
                else:
                    latent_idx = self.frame_dict[frame_idx]
                latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
                latent_code = self.latent(latent_idx)
                latent_code = latent_code.expand(pose_feat.shape[0], -1)
                pose_feat = torch.cat([pose_feat, latent_code], dim=1)
        else:
            bone_transforms = camera.bone_transforms
            bone_transform_flatten = torch.flatten(bone_transforms, start_dim=1)
            bone_transform_flatten = bone_transform_flatten.view(-1, 384)
            bone_enc = self.bone_encoder(bone_transform_flatten)


            xyz = self.vb_model.get_virtual_joints()    
            xyz_norm = self.aabb.normalize(xyz, sym=True)
            feature = self.hashgrid(xyz_norm)
            bone_cond = {'smpl': bone_enc, 'hashgrid': feature} 
        garm_label = 1
        lbs_network = False
        if return_mat:
            xyz = gaussians.get_xyz_by_category(garm_label, self.trainable_label) 
            n_pts = xyz.shape[0]
            if lbs_network:
                virtual_weights = self.vb_lbs_network(xyz)
            else:
                virtual_weights = self.weighted_knn(xyz) # size of (N, num_vb)
            # vb_rotation, vb_translation = self.get_vb_deformation(self.vb_model.get_virtual_joints(), bone_cond)
            if self.non_rigid_style:
                vb_translation, _, vb_rotation = self.get_mlp_deformation(self.vb_model.get_virtual_joints(), pose_feat)
            else:
                vb_rotation, vb_translation = self.get_vb_deformation(self.vb_model.get_virtual_joints(), bone_cond)
            vb_rotation, quat = batch_rodrigues(vb_rotation) # from axis-angle to rotation matrix

            
            # compute the weighted deformation
            weighted_rotation = torch.matmul(virtual_weights, vb_rotation.reshape(-1,9)).squeeze(1)
            weighted_rotation = torch.matmul(weighted_rotation.view(n_pts, 3, 3)[:,:3,:], camera.root_orient_mat.cuda()).view(n_pts, 9)
            # weighted_rotation = gram_schmidt_batch(weighted_rotation.view(n_pts, 3, 3)).view(n_pts, 9) too slow
            weighted_translation = torch.matmul(virtual_weights, vb_translation.squeeze()) + torch.tensor(camera.transl).cuda()
            

            T_fwd = torch.cat((weighted_rotation.view(-1,3,3), weighted_translation.unsqueeze(2)), dim=-1)
            T_fwd = torch.cat((T_fwd, torch.tensor([0,0,0,1]).cuda().repeat(n_pts,1).unsqueeze(1)),dim=1)

            deformed_gaussians = gaussians.clone()
            deformed_gaussians.set_fwd_transform(gaussians.get_fwd_transform().clone())
            deformed_gaussians.set_fwd_transform_by_category(garm_label, T_fwd.clone(), self.trainable_label)
            # T_fwd = deformed_gaussians.get_fwd_transform_by_category(garm_label)    

            homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
            x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
            x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
            # deformed_gaussians._xyz = x_bar
            deformed_gaussians.set_xyz_by_category(garm_label, x_bar, self.trainable_label)

            # rotation_hat = build_rotation(gaussians._rotation)
            if self.trainable_label:
                rotation_hat = build_rotation(gaussians._rotation)
                rotation_bar = rotation_hat.clone()
                rotation_hat_garm = rotation_hat[deformed_gaussians._label_trainable[:, 0] >= 0.5]
                rotation_bar_garm = torch.matmul(T_fwd[:, :3, :3], rotation_hat_garm)
                rotation_bar[deformed_gaussians._label_trainable[:, 0] >= 0.5] = rotation_bar_garm
                setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
            else:                
                rotation_hat = build_rotation(gaussians._rotation)
                rotation_bar = rotation_hat.clone()
                rotation_hat_garm = rotation_hat[deformed_gaussians._label[:, 0] == garm_label]
                rotation_bar_garm = torch.matmul(T_fwd[:, :3, :3], rotation_hat_garm)
                rotation_bar[deformed_gaussians._label[:, 0] == garm_label] = rotation_bar_garm
                setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)

            cano_garm = gaussians.get_xyz_by_category(garm_label, self.trainable_label).detach()
            deformed_garm = deformed_gaussians.get_xyz_by_category(garm_label, self.trainable_label).detach()
            if self.save_deform:
                if iteration % 500 == 0 and iteration > 8000 and iteration < 15000:
                    save_ply_garmweights(cano_garm, deformed_garm, self.vb_model.get_virtual_joints(), virtual_weights, iteration, self.dir_ply)

            # nodes_deformed = self.forward_graph(nodes=self.virtual_joints, cond=bone_cond, smpl_tfs=bone_transforms, smpl_root_orient=camera.root_orient_mat.cuda(), smpl_trans=camera.transl, scale=None,
            #           time_enc=time_enc)
            # nodes_deformed = self.forward_graph(nodes=self.vb_model.get_virtual_joints(), cond=bone_cond, smpl_tfs=bone_transforms, smpl_root_orient=camera.root_orient_mat.cuda(), smpl_trans=camera.transl, scale=None,
            #           time_enc=time_enc)
            nodes_deformed = None
            return deformed_gaussians, nodes_deformed
        
        else:
            # nodes_deformed = self.forward_graph(nodes=self.virtual_joints, cond=bone_cond, smpl_tfs=bone_transforms, smpl_root_orient=camera.root_orient_mat.cuda(), smpl_trans=camera.transl, scale=None,
            #           time_enc=time_enc)
            nodes_deformed = self.forward_graph(nodes=self.vb_model.get_virtual_bones, cond=bone_cond, smpl_tfs=bone_transforms, smpl_root_orient=camera.root_orient_mat.cuda(), smpl_trans=camera.transl, scale=None,
                      time_enc=time_enc)
            xyz = gaussians.get_xyz_by_category(garm_label, self.trainable_label)

            n_pts = xyz.shape[0]
            virtual_weights = self.weighted_knn(xyz) # size of (N, num_vb)
            deformed_xyz = torch.matmul(virtual_weights, nodes_deformed)

            deformed_gaussians = gaussians.clone()
            deformed_gaussians.set_fwd_transform(gaussians.get_fwd_transform().detach())
            return deformed_gaussians, nodes_deformed

            


    def get_transformation(self, nodes, cond, time_enc):
        transformation = self.deformation_graph(nodes, cond, time_enc) # predict the 6 DoF
        rot = transformation[:, :, :3] # TODO try to use the quaternion representation
        trans = transformation[0, :, 3:].unsqueeze(-1)

        rot_mat, quat = batch_rodrigues(rot[0]) # from axis-angle to rotation matrix
        # rot_mat = euler2rotmat(rot)
        transform_mat = to_transform_mat(rot_mat, trans)
        return transform_mat
    
    def update_deformation_nodes(self, deformation_graph_verts):
        if not torch.is_tensor(deformation_graph_verts):
            self.deformation_graph_verts = torch.from_numpy(deformation_graph_verts).float().cuda().detach()
        else:
            assert False, "deformation_graph_verts should be a numpy array"
            self.deformation_graph_verts = deformation_graph_verts

    def get_pts_W(self, gaussians):
        garm_label = 1
        xyz = gaussians.get_xyz_by_category(garm_label, self.trainable_label)
        n_pts = xyz.shape[0]



    # def softmax(self, logit):
    #     # if logit.shape[-1] == 25:
    #     if logit.shape[-1] == self.d_out:
    #         w = hierarchical_softmax(logit)
    #         # w = F.softmax(logit, dim=-1)
    #     # elif logit.shape[-1] == 24:
    #     elif logit.shape[-1] != self.d_out:
    #         w = F.softmax(logit, dim=-1)
    #     else:
    #         raise ValueError
    #     return w

    # # get (N, J) representing the joint weights of each point
    # def get_xyz_J(self, gaussians):
    #     xyz = gaussians.get_xyz
    #     xyz_norm = self.aabb.normalize(xyz, sym=True)
    #     knn_weights = self.query_weights(xyz_norm)
    #     pts_W = self.softmax(self.lbs_network(xyz_norm))

    #     # pts_W = self.lambda_knn_res * knn_weights + (1 - self.lambda_knn_res) * pts_W
    #     return self.lambda_knn_res * knn_weights + (1 - self.lambda_knn_res) * pts_W
    
    def query_weights(self, xyz):
        # find the nearest vertex
        # knn_ret = ops.knn_points(xyz.unsqueeze(0), self.virtual_joints.unsqueeze(0))
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.vb_model.get_virtual_joints().unsqueeze(0)) 
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights_tensor[p_idx, :]
        #Todo: decide a correct skinning weight
        return pts_W

    def weighted_knn(self, xyz):

        K = 5
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.vb_model.get_virtual_joints().unsqueeze(0), K=K)
        p_idx = knn_ret.idx.squeeze(0)
        p_dists = knn_ret.dists.squeeze(0)
        p_dists = 1 / (p_dists + 1e-8)

        # weighted_knn = return an array of N * xyz[0], the distance of the N virtual joints with respect to the i-th xyz, only the
        # nearest k positions set to non-zero, the valid k position should be normalized
        weighted_knn = torch.zeros(xyz.shape[0], self.num_vb, device=xyz.device)
        weighted_knn.scatter_(1, p_idx, p_dists)
        row_sum = torch.sum(weighted_knn, dim=1)
        weighted_knn = torch.transpose(weighted_knn,0,1) / row_sum
        weighted_knn = torch.transpose(weighted_knn,0,1)
        return weighted_knn

        # distance_squared, nn_index, _ = ops.knn_points(xyz.unsqueeze(0), self.vb_model.get_virtual_joints().unsqueeze(0), K=K, return_nn=False)
        # distance = torch.sqrt(distance_squared)
        # distance = torch.sqrt(distance_squared)
        # distance = torch.clamp(distance, max=1)
        # weights = -torch.log(distance - 1e-6)[0]
        # weights = weights / weights.sum(dim=-1, keepdim=True)

        # skinning_weights = torch.zeros((xyz.shape[0], self.num_vb)).cuda() # TODO sparse matrix is better
        # skinning_weights.scatter_(1, nn_index[0], weights) # TODO double-check scatter_
        # return skinning_weights
    
    # concatenate the nodes and time_embedding, producing the rotation and translation for all virtual bones
    def get_vb_deformation(self, nodes, cond, time_enc=None):
        # # version producing a rotation matrix and translation vector
        # transform_mat = self.deformation_graph(nodes, cond, time_enc)      # 9+3 DOF
        # import ipdb; ipdb.set_trace()
        # return transform_mat[:,:9], transform_mat[:,9:]

        # version producing rotation angle
        transformation = self.deformation_graph(nodes, cond, time_enc) # predict the 6 DoF
        rot = transformation[:, :3]  
        trans = transformation[:, 3:].unsqueeze(-1)
        return rot, trans
    
    def get_mlp_deformation(self, nodes, cond):
        deltas = self.mlp(nodes, cond=cond)
        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]
        return delta_xyz, delta_scale, delta_rot
    # return the deformation for each virtual joints
    def forward_graph(self, nodes=None, cond=None, smpl_tfs=None, smpl_root_orient=None, smpl_trans=None, scale=None,
                      time_enc=None):
        rot, trans = self.get_vb_deformation(nodes, cond)
        rot_mat, quat = batch_rodrigues(rot) # from axis-angle to rotation matrix
        transform_mat = to_transform_mat(rot_mat, trans).unsqueeze(0)
        
        # Version 1
        # smpl_root_orient = smpl_tfs[:, 0:1]
        # transform_mat = smpl_root_orient @ transform_mat
        
        # Version 2
        # smpl_root_orient_mat = batch_rodrigues(smpl_root_orient)
        # smpl_root_orient_mat = smpl_root_orient.unsqueeze(0)
        # smpl_root_orient_mat = to_transform_mat(smpl_root_orient_mat, torch.zeros([smpl_root_orient_mat.shape[0], 3, 1]).cuda()).unsqueeze(0).detach()
        # transform_mat = torch.matmul(smpl_root_orient_mat.expand(-1, transform_mat.shape[1], -1, -1), transform_mat).squeeze(0)
        # transform_mat[:, :3, 3] = transform_mat[:, :3, 3] + torch.tensor(smpl_trans).cuda().unsqueeze(0)
        
        transform_mat = transform_mat.squeeze(0)
        skinning_weights_self = torch.eye(nodes.shape[0]).cuda()
        nodes_deformed = skinning(nodes[None], skinning_weights_self[None], transform_mat.unsqueeze(0), inverse=False, return_T=False)
        return nodes_deformed.squeeze(0)


    def get_forward_transform(self, nodes, cond, time_enc):
        transformation = self.deformation_graph(nodes, cond, time_enc) # predict the 6 DoF
        rot = transformation[:, :, :3]
        trans = transformation[0, :, 3:].unsqueeze(-1)
        rot_mat, quat = batch_rodrigues(rot[0]) # from axis-angle to rotation matrix
        transform_mat = to_transform_mat(rot_mat, trans)
        return transform_mat
    
    def get_virtual_joints(self):
        # return self.virtual_joints
        return self.vb_model.get_virtual_joints()

    def regularization(self):
        loss_skinning, pts_skinning, sampled_weights, pred_weights = self.get_skinning_loss()
        return {
            'loss_skinning': loss_skinning,
            'pts_skinning': pts_skinning,
            'sampled_weights': sampled_weights,
            'pred_weights': pred_weights
        }
    
    def extract_virtual_bones(self, gaussians):
        # virtual_bones = gaussians.extract_virtual_bones()
        # self.virtual_joints = virtual_bones
        # self.num_vb = self.virtual_joints.shape[0]
        # return self.virtual_joints
        self.num_vb = 80
        virtual_joints = self.vb_model.extract_virtual_bones(gaussians, self.num_vb, self.vb_xyz_trainable)
        # virtual_joints = self.vb_model.extract_virtual_bones(gaussians)
        self.num_vb  = virtual_joints.shape[0]
        self.vb_lbs_network = get_skinning_mlp(3, self.num_vb, self.cfg.skinning_network).cuda()
        return virtual_joints
    
class DeformationGraph_res(Garm_Simulator):
    def __init__(self, cfg, metadata, garm_simulator_args):
        super().__init__(cfg)
        self.vb_mode = garm_simulator_args["vb_mode"]
        self.vb_delay = garm_simulator_args["vb_delay"]

        if self.vb_mode == 'enable':
            self.virtual_bones = metadata["virtual_bones"]
            self.virtual_joints = torch.tensor(self.virtual_bones.vertices).float().cuda()
            self.num_vb = self.virtual_joints.shape[0]
        self.aabb = metadata["aabb"]
        # self.root_orient = metadata["root_orient"]
        # self.trans = metadata["trans"]
        self.update_vb = cfg.update_vb
        self.non_rigid_style = True

        self.trainable_label = garm_simulator_args["trainable_label"]  
        self.vb_model = VirtualBone(metadata['cameras_extent'])
        self.vb_xyz_trainable = garm_simulator_args["vb_xyz_trainable"]   
        self.save_deform = garm_simulator_args["save_deform"]
        self.dir_ply = "assets/garm_debug/{0}".format(garm_simulator_args["dir_save_ply"])

        self.pose_encoder = HierarchicalPoseEncoder(**cfg.pose_encoder)
        d_cond = self.pose_encoder.n_output_dims
        # add latent code
        self.latent_dim = cfg.get('latent_dim', 0)
        if self.latent_dim > 0:
            d_cond += self.latent_dim
            self.frame_dict = metadata['frame_dict']
            self.latent = nn.Embedding(len(self.frame_dict), self.latent_dim)
        d_in = 3
        d_out = 3 + 3 + 3
        self.feature_dim = cfg.get('feature_dim', 0)
        d_out += self.feature_dim

        # output dimension: position + scale + rotation
        self.mlp = VanillaCondMLP(d_in, d_cond, d_out, cfg.mlp)

    def forward(self, gaussians, iteration, camera, time_enc, **kwargs):
        # # pose encoding
        rots = camera.rots
        Jtrs = camera.Jtrs
        pose_feat = self.pose_encoder(rots, Jtrs)

        if self.latent_dim > 0:
            frame_idx = camera.frame_id
            if frame_idx not in self.frame_dict:
                latent_idx = len(self.frame_dict) - 1
            else:
                latent_idx = self.frame_dict[frame_idx]
            latent_idx = torch.Tensor([latent_idx]).long().to(pose_feat.device)
            latent_code = self.latent(latent_idx)
            latent_code = latent_code.expand(pose_feat.shape[0], -1)
            pose_feat = torch.cat([pose_feat, latent_code], dim=1)
        garm_label = 1

        xyz = gaussians.get_xyz_by_category(garm_label, self.trainable_label) 
        n_pts = xyz.shape[0]

        virtual_weights = self.weighted_knn(xyz) # size of (N, num_vb)
        # vb_rotation, vb_translation = self.get_vb_deformation(self.vb_model.get_virtual_joints(), bone_cond)
        vb_translation, _, vb_rotation = self.get_mlp_deformation(self.vb_model.get_virtual_joints(), pose_feat)
        vb_rotation, quat = batch_rodrigues(vb_rotation) # from axis-angle to rotation matrix

        
        # compute the weighted deformation
        weighted_rotation = torch.matmul(virtual_weights, vb_rotation.reshape(-1,9)).squeeze(1)
        # weighted_rotation = gram_schmidt_batch(weighted_rotation.view(n_pts, 3, 3)).view(n_pts, 9) too slow
        weighted_translation = torch.matmul(virtual_weights, vb_translation.squeeze())
        

        T_fwd = torch.cat((weighted_rotation.view(-1,3,3), weighted_translation.unsqueeze(2)), dim=-1)
        T_fwd = torch.cat((T_fwd, torch.tensor([0,0,0,1]).cuda().repeat(n_pts,1).unsqueeze(1)),dim=1)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(gaussians.get_fwd_transform().clone())
        deformed_gaussians.set_fwd_transform_by_category(garm_label, T_fwd.clone(), self.trainable_label)
        # T_fwd = deformed_gaussians.get_fwd_transform_by_category(garm_label)    

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        # deformed_gaussians._xyz = x_bar
        deformed_gaussians.set_xyz_by_category(garm_label, x_bar, self.trainable_label)

        # rotation_hat = build_rotation(gaussians._rotation)
        if self.trainable_label:
            rotation_hat = build_rotation(gaussians._rotation)
            rotation_bar = rotation_hat.clone()
            rotation_hat_garm = rotation_hat[deformed_gaussians._label_trainable[:, 0] >= 0.5]
            rotation_bar_garm = torch.matmul(T_fwd[:, :3, :3], rotation_hat_garm)
            rotation_bar[deformed_gaussians._label_trainable[:, 0] >= 0.5] = rotation_bar_garm
            setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        else:                
            rotation_hat = build_rotation(gaussians._rotation)
            rotation_bar = rotation_hat.clone()
            rotation_hat_garm = rotation_hat[deformed_gaussians._label[:, 0] == garm_label]
            rotation_bar_garm = torch.matmul(T_fwd[:, :3, :3], rotation_hat_garm)
            rotation_bar[deformed_gaussians._label[:, 0] == garm_label] = rotation_bar_garm
            setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)

        cano_garm = gaussians.get_xyz_by_category(garm_label, self.trainable_label).detach()
        deformed_garm = deformed_gaussians.get_xyz_by_category(garm_label, self.trainable_label).detach()
        if self.save_deform:
            if iteration % 500 == 0 and iteration > 8000 and iteration < 15000:
                save_ply_garmweights(cano_garm, deformed_garm, self.vb_model.get_virtual_joints(), virtual_weights, iteration, self.dir_ply)

        # nodes_deformed = self.forward_graph(nodes=self.virtual_joints, cond=bone_cond, smpl_tfs=bone_transforms, smpl_root_orient=camera.root_orient_mat.cuda(), smpl_trans=camera.transl, scale=None,
        #           time_enc=time_enc)
        # nodes_deformed = self.forward_graph(nodes=self.vb_model.get_virtual_joints(), cond=bone_cond, smpl_tfs=bone_transforms, smpl_root_orient=camera.root_orient_mat.cuda(), smpl_trans=camera.transl, scale=None,
        #           time_enc=time_enc)
        nodes_deformed = None
        return deformed_gaussians, nodes_deformed
    
    def get_mlp_deformation(self, nodes, cond):
        deltas = self.mlp(nodes, cond=cond)
        delta_xyz = deltas[:, :3]
        delta_scale = deltas[:, 3:6]
        delta_rot = deltas[:, 6:10]
        return delta_xyz, delta_scale, delta_rot
    
    def weighted_knn(self, xyz):
        K = 5
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.vb_model.get_virtual_joints().unsqueeze(0), K=K)
        p_idx = knn_ret.idx.squeeze(0)
        p_dists = knn_ret.dists.squeeze(0)
        p_dists = 1 / (p_dists + 1e-8)

        # weighted_knn = return an array of N * xyz[0], the distance of the N virtual joints with respect to the i-th xyz, only the
        # nearest k positions set to non-zero, the valid k position should be normalized
        weighted_knn = torch.zeros(xyz.shape[0], self.num_vb, device=xyz.device)
        weighted_knn.scatter_(1, p_idx, p_dists)
        row_sum = torch.sum(weighted_knn, dim=1)
        weighted_knn = torch.transpose(weighted_knn,0,1) / row_sum
        weighted_knn = torch.transpose(weighted_knn,0,1)
        return weighted_knn
    
    def extract_virtual_bones(self, gaussians):
        self.num_vb = 80
        virtual_joints = self.vb_model.extract_virtual_bones(gaussians, self.num_vb, self.vb_xyz_trainable)
        self.num_vb  = virtual_joints.shape[0]
        self.vb_lbs_network = get_skinning_mlp(3, self.num_vb, self.cfg.skinning_network).cuda()
        return virtual_joints
    
    def get_virtual_joints(self):
        # return self.virtual_joints
        return self.vb_model.get_virtual_joints()

def batch_rodrigues(axis_ang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axis_ang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axis_ang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    # rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat, quat

def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def to_transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def get_garm_simulator(cfg, metadata, garm_simulator_args):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "deformation_graph": DeformationGraph,
        "deformation_graph_res": DeformationGraph_res   
    }
    return model_dict[name](cfg, metadata, garm_simulator_args)

def skinning(x, w, tfs, inverse=False, return_T=False):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    x_h = F.pad(x, (0, 1), value=1.0)

    if inverse:
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)
    else:
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)
        w_tf = None
    if return_T:
        return x_h[:, :, :3], w_tf
    else:
        return x_h[:, :, :3]
    

def save_ply_garmweights(garm_xyz, deformed_xyz, virtual_joints, pred_weights, iteration, save_dir):
    num_vb = virtual_joints.shape[0]
    cmap = plt.cm.get_cmap('tab20', num_vb)
    joint_colors = torch.tensor([cmap(i)[:3] for i in range(num_vb)])
    pred = pred_weights.detach().cpu()
    maxjoint_pred_idx = torch.argmax(pred, dim=1)
    pred_colors = joint_colors[maxjoint_pred_idx]
    file_garm = save_dir + "/cano_garm_{0}.ply".format(iteration)
    file_deformed = save_dir + "/deformed_garm_{0}.ply".format(iteration)    
    file_vjoints = save_dir + "/virtual_joints_{0}.ply".format(iteration)
    vert2hetereoply(garm_xyz, file_garm, pred_colors)
    vert2hetereoply(deformed_xyz, file_deformed, pred_colors)
    vert2hetereoply(virtual_joints, file_vjoints, joint_colors)