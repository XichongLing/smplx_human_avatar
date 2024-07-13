import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d.ops as ops
import trimesh
import igl

from utils.general_utils import build_rotation
from models.network_utils import get_skinning_mlp
from utils.dataset_utils import AABB 

import copy

import matplotlib.pyplot as plt
class RigidDeform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, gaussians, iteration, camera, **kwargs):
        raise NotImplementedError

    def regularization(self):
        return NotImplementedError
    
    def get_xyz_J(self, gaussians):
        return torch.empty(0)


class Identity(RigidDeform):
    """ Identity mapping for single frame reconstruction """
    def __init__(self, cfg, metadata):
        super().__init__(cfg)

    def forward(self, gaussians, iteration, camera, **kwargs):
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

class SMPLNN(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.smpl_verts = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
        self.skinning_weights = torch.from_numpy(metadata["skinning_weights"]).float().cuda()

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights[p_idx, :]

        return pts_W

    def forward(self, gaussians, iteration, camera, **kwargs):
        bone_transforms = camera.bone_transforms

        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        pts_W = self.query_weights(xyz)
        T_fwd = torch.matmul(pts_W, bone_transforms.view(-1, 16)).view(n_pts, 4, 4).float()

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians, torch.empty(0)

    def regularization(self):
        return {}

def create_voxel_grid(d, h, w, device='cpu'):
    x_range = (torch.linspace(-1,1,steps=w,device=device)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
    y_range = (torch.linspace(-1,1,steps=h,device=device)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
    z_range = (torch.linspace(-1,1,steps=d,device=device)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
    grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3,-1).permute(0,2,1)

    return grid

''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return F.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_point, n_dim = x.shape

    if n_dim == 25:
        prob_all = torch.ones(n_point, 24, device=x.device)
        # softmax_x = F.softmax(x, dim=-1)
        sigmoid_x = sigmoid(x).float()

        prob_all[:, [1, 2, 3]] = sigmoid_x[:, [0]] * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = 1 - sigmoid_x[:, [0]]

        prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid_x[:, [4, 5, 6]])
        prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid_x[:, [4, 5, 6]])

        prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid_x[:, [7, 8, 9]])
        prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid_x[:, [7, 8, 9]])

        prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid_x[:, [10, 11]])
        prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid_x[:, [10, 11]])

        prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid_x[:, [24]] * softmax(x[:, [12, 13, 14]])
        prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid_x[:, [24]])

        prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid_x[:, [15]])
        prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid_x[:, [15]])

        prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid_x[:, [16, 17]])
        prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid_x[:, [16, 17]])

        prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid_x[:, [18, 19]])
        prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid_x[:, [18, 19]])

        prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid_x[:, [20, 21]])
        prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid_x[:, [20, 21]])

        prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid_x[:, [22, 23]])
        prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid_x[:, [22, 23]])
    else:
        prob_all = torch.ones(n_point, 55, device=x.device)
        # softmax_x = F.softmax(x, dim=-1)
        sigmoid_x = sigmoid(x).float()

        prob_all[:, [1, 2, 3]] = sigmoid_x[:, [0]] * softmax(x[:, [1, 2, 3]])
        prob_all[:, [0]] = 1 - sigmoid_x[:, [0]]

        prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid_x[:, [4, 5, 6]])
        prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid_x[:, [4, 5, 6]])

        prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid_x[:, [7, 8, 9]])
        prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid_x[:, [7, 8, 9]])

        prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid_x[:, [10, 11]])
        prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid_x[:, [10, 11]])

        prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid_x[:, [55]] * softmax(x[:, [12, 13, 14]])
        prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid_x[:, [55]])

        prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid_x[:, [15]])
        prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid_x[:, [15]])

        prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid_x[:, [16, 17]])
        prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid_x[:, [16, 17]])

        prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid_x[:, [18, 19]])
        prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid_x[:, [18, 19]])

        prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid_x[:, [20, 21]])
        prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid_x[:, [20, 21]])

        prob_all[:, [22, 23, 24]] = prob_all[:, [15]] * sigmoid_x[:, [56]] * softmax(x[:, [22, 23, 24]])
        prob_all[:, [15]] = prob_all[:, [15]] * (1 - sigmoid_x[:, [56]])

        prob_all[:, [25, 28, 31, 34, 37]] = prob_all[:, [20]] * sigmoid_x[:, [57]] * softmax(x[:, [25, 28, 31, 34, 37]])
        prob_all[:, [20]] = prob_all[:, [20]] * (1 - sigmoid_x[:, [57]])

        prob_all[:, [26, 29, 32, 35, 38]] = prob_all[:, [25, 28, 31, 34, 37]] * (sigmoid_x[:, [26, 29, 32, 35, 38]])
        prob_all[:, [25, 28, 31, 34, 37]] = prob_all[:, [25, 28, 31, 34, 37]] * (1 - sigmoid_x[:, [26, 29, 32, 35, 38]])

        prob_all[:, [27, 30, 33, 36, 39]] = prob_all[:, [26, 29, 32, 35, 38]] * (sigmoid_x[:, [27, 30, 33, 36, 39]])
        prob_all[:, [26, 29, 32, 35, 38]] = prob_all[:, [26, 29, 32, 35, 38]] * (1 - sigmoid_x[:, [27, 30, 33, 36, 39]])

        prob_all[:, [40, 43, 46, 49, 52]] = prob_all[:, [21]] * sigmoid_x[:, [58]] * softmax(x[:, [40, 43, 46, 49, 52]])
        prob_all[:, [21]] = prob_all[:, [21]] * (1 - sigmoid_x[:, [58]])

        prob_all[:, [41, 44, 47, 50, 53]] = prob_all[:, [40, 43, 46, 49, 52]] * (sigmoid_x[:, [41, 44, 47, 50, 53]])
        prob_all[:, [40, 43, 46, 49, 52]] = prob_all[:, [40, 43, 46, 49, 52]] * (1 - sigmoid_x[:, [41, 44, 47, 50, 53]])

        prob_all[:, [42, 45, 48, 51, 54]] = prob_all[:, [41, 44, 47, 50, 53]] * (sigmoid_x[:, [42, 45, 48, 51, 54]])
        prob_all[:, [41, 44, 47, 50, 53]] = prob_all[:, [41, 44, 47, 50, 53]] * (1 - sigmoid_x[:, [42, 45, 48, 51, 54]])

        # prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all

class SkinningField(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.smpl_verts = metadata["smpl_verts"]
        self.skinning_weights = metadata["skinning_weights"]
        
        self.smpl_verts_tensor = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
        self.skinning_weights_tensor = torch.from_numpy(metadata["skinning_weights"]).float().cuda()
        
        
        self.aabb = metadata["aabb"]
        self.cano_aabb = copy.deepcopy(metadata['aabb'])

        # self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.faces = metadata['faces']
        self.cano_mesh = metadata["cano_mesh"]
  

        self.distill = cfg.distill
        d, h, w = cfg.res // cfg.z_ratio, cfg.res, cfg.res
        self.resolution = (d, h, w)
        if self.distill:
            self.grid = create_voxel_grid(d, h, w).cuda()

        self.lambda_knn_res = cfg.lambda_knn_res

        self.lbs_network = get_skinning_mlp(3, 25 , cfg.skinning_network)
        # need to further check d_out is the num_joint?
        # - theoretically yes, lbs_network input: sampled cano points, output: weights for each joint
        self.d_out = cfg.d_out


    def precompute(self, recompute_skinning=True):
        if recompute_skinning or not hasattr(self, "lbs_voxel_final"):
            d, h, w = self.resolution

            lbs_voxel_final = self.lbs_network(self.grid[0]).float()
            lbs_voxel_final = self.cfg.soft_blend * lbs_voxel_final

            lbs_voxel_final = self.softmax(lbs_voxel_final)

            self.lbs_voxel_final = lbs_voxel_final.permute(1, 0).reshape(1, 24, d, h, w)

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts_tensor.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights_tensor[p_idx, :]
        return pts_W
    
    def get_forward_transform(self, xyz, tfs):
        if self.distill:
            self.precompute(recompute_skinning=self.training)
            fwd_grid = torch.einsum("bcdhw,bcxy->bxydhw", self.lbs_voxel_final, tfs[None])
            fwd_grid = fwd_grid.reshape(1, -1, *self.resolution)
            T_fwd = F.grid_sample(fwd_grid, xyz.reshape(1, 1, 1, -1, 3), padding_mode='border')
            T_fwd = T_fwd.reshape(4, 4, -1).permute(2, 0, 1)
        else:
            pts_W = self.softmax(self.lbs_network(xyz))
            # pred_weights = self.lambda_knn_res * knn_weight + (1 - self.lambda_knn_res) * self.softmax(self.lbs_network(pts_skinning))
            # import ipdb; ipdb.set_trace()
            T_fwd = torch.matmul(pts_W, tfs.view(-1, 16)).view(-1, 4, 4).float()
        return T_fwd, pts_W
    
    def save_canonical_weights(self, path):
        points_skinning, face_idx = self.cano_mesh.sample(self.cfg.n_reg_pts * 3, return_index=True)
        points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
        bary_coords = igl.barycentric_coordinates_tri(
            points_skinning,
            self.smpl_verts[self.faces[face_idx, 0], :],
            self.smpl_verts[self.faces[face_idx, 1], :],
            self.smpl_verts[self.faces[face_idx, 2], :],
        )
        vert_ids = self.faces[face_idx, ...]
        pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)
        # adding samples for hand


        points_skinning = torch.from_numpy(points_skinning).cuda()
        pts_W = torch.from_numpy(pts_W).cuda()

        # normalize it
        points_skinning = self.cano_aabb.normalize(points_skinning, sym=True)


        joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
        joint_colors = joint_colors[:,:3]
        points = points_skinning.detach().cpu()
        pred = pts_W.detach().cpu()
        maxjoint_pred_idx = torch.argmax(pred, dim=1)
        pred_colors = joint_colors[maxjoint_pred_idx]
        with open(path, 'w') as f:
            for point, color in zip(points, pred_colors):
                # Write XYZRGB data to the file
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        print("XYZRGB file created successfully.")

        return points_skinning, pts_W

    def sample_skinning_loss(self):
        points_skinning, face_idx = self.cano_mesh.sample(self.cfg.n_reg_pts, return_index=True)
        points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
        bary_coords = igl.barycentric_coordinates_tri(
            points_skinning,
            self.smpl_verts[self.faces[face_idx, 0], :],
            self.smpl_verts[self.faces[face_idx, 1], :],
            self.smpl_verts[self.faces[face_idx, 2], :],
        )
        vert_ids = self.faces[face_idx, ...]
        pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)

        points_skinning = torch.from_numpy(points_skinning).cuda()
        pts_W = torch.from_numpy(pts_W).cuda()

        return points_skinning, pts_W

    def softmax(self, logit):
        # if logit.shape[-1] == 25:
        if logit.shape[-1] == self.d_out:
            w = hierarchical_softmax(logit)
            # w = F.softmax(logit, dim=-1)
        # elif logit.shape[-1] == 24:
        elif logit.shape[-1] != self.d_out:
            w = F.softmax(logit, dim=-1)
        else:
            raise ValueError
        return w

    def get_skinning_loss(self):
        pts_skinning, sampled_weights = self.sample_skinning_loss()
        pts_skinning = self.cano_aabb.normalize(pts_skinning, sym=True)

        
        knn_weight = self.query_weights(pts_skinning)

        if self.distill:
            pred_weights = F.grid_sample(self.lbs_voxel_final, pts_skinning.reshape(1, 1, 1, -1, 3), padding_mode='border')
            pred_weights = pred_weights.reshape(24, -1).permute(1, 0)
        else:
            pred_weights = self.lambda_knn_res * knn_weight + (1 - self.lambda_knn_res) * self.softmax(self.lbs_network(pts_skinning))
        skinning_loss = torch.nn.functional.mse_loss(
            pred_weights, sampled_weights, reduction='none').sum(-1).mean()

        # try entropy loss
        # skinning_loss = 0.2 * torch.nn.functional.cross_entropy(pred_weights, sampled_weights).mean()

        return skinning_loss, pts_skinning, sampled_weights, pred_weights
    
    # get (N, J) representing the joint weights of each point
    def get_xyz_J(self, gaussians):
        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        knn_weights = self.query_weights(xyz_norm)
        pts_W = self.softmax(self.lbs_network(xyz_norm))

        # pts_W = self.lambda_knn_res * knn_weights + (1 - self.lambda_knn_res) * pts_W
        return self.lambda_knn_res * knn_weights + (1 - self.lambda_knn_res) * pts_W



    def forward(self, gaussians, iteration, camera):
        tfs = camera.bone_transforms
        # Gaussian position
        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]

        # if iteration < 6000 and iteration %2000 == 0:
        #     coord_max = np.max(xyz.detach().cpu().numpy(), axis=0)
        #     coord_min = np.min(xyz.detach().cpu().numpy(), axis=0)
        #     # hard code the padding as 0.1 here
        #     # later should be a parameter
        #     padding_ratio = 0.1
        #     padding_ratio = np.array(padding_ratio, dtype=np.float32)
        #     padding = (coord_max - coord_min) * padding_ratio
        #     coord_max += padding 
        #     coord_min -= padding

        #     self.aabb.update(coord_max, coord_min)
        # normalizing position 
        xyz_norm = self.aabb.normalize(xyz, sym=True)

        T_fwd, pts_W = self.get_forward_transform(xyz_norm, tfs)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians, pts_W

    def regularization(self):
        loss_skinning, pts_skinning, sampled_weights, pred_weights = self.get_skinning_loss()
        return {
            'loss_skinning': loss_skinning,
            'pts_skinning': pts_skinning,
            'sampled_weights': sampled_weights,
            'pred_weights': pred_weights
        }
# need to change here
class SkinningFieldSmplx(RigidDeform):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.smpl_verts = metadata["smpl_verts"]
        self.skinning_weights = metadata["skinning_weights"]
        
        self.smpl_verts_tensor = torch.from_numpy(metadata["smpl_verts"]).float().cuda()
        self.skinning_weights_tensor = torch.from_numpy(metadata["skinning_weights"]).float().cuda()
        
        
        self.aabb = metadata["aabb"]
        self.cano_aabb = copy.deepcopy(metadata['aabb'])
        # self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.faces = metadata['faces']
        self.cano_mesh = metadata["cano_mesh"]
        self.cano_hand_mesh = metadata["cano_hand_mesh"]
        self.hand2cano = metadata['hand2cano_dict']

        self.distill = cfg.distill
        d, h, w = cfg.res // cfg.z_ratio, cfg.res, cfg.res
        self.resolution = (d, h, w)
        if self.distill:
            self.grid = create_voxel_grid(d, h, w).cuda()

        self.lambda_knn_res = cfg.lambda_knn_res

        self.lbs_network = get_skinning_mlp(3, 55 , cfg.skinning_network)
        # self.lbs_network = Smplx_lbs(d_out= 55, cfg=cfg)
        # need to further check d_out is the num_joint?
        # wrist part: 21 22
        # - theoretically yes, lbs_network input: sampled cano points, output: weights for each joint
        self.d_out = cfg.d_out


    def precompute(self, recompute_skinning=True):
        if recompute_skinning or not hasattr(self, "lbs_voxel_final"):
            d, h, w = self.resolution

            lbs_voxel_final = self.lbs_network(self.grid[0]).float()
            lbs_voxel_final = self.cfg.soft_blend * lbs_voxel_final

            lbs_voxel_final = self.softmax(lbs_voxel_final)

            self.lbs_voxel_final = lbs_voxel_final.permute(1, 0).reshape(1, 24, d, h, w)

    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.smpl_verts_tensor.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights_tensor[p_idx, :]
        return pts_W
    
    def get_forward_transform(self, xyz, tfs):
        if self.distill:
            self.precompute(recompute_skinning=self.training)
            fwd_grid = torch.einsum("bcdhw,bcxy->bxydhw", self.lbs_voxel_final, tfs[None])
            fwd_grid = fwd_grid.reshape(1, -1, *self.resolution)
            T_fwd = F.grid_sample(fwd_grid, xyz.reshape(1, 1, 1, -1, 3), padding_mode='border')
            T_fwd = T_fwd.reshape(4, 4, -1).permute(2, 0, 1)
        else:
            # try resiudal connection
            # hardcode the lambda now, dont foget to change get skinning loss as well
            knn_weight = self.query_weights(xyz)

            pts_W = self.lambda_knn_res * knn_weight + (1 - self.lambda_knn_res) * self.softmax(self.lbs_network(xyz))

            # import ipdb; ipdb.set_trace()
            T_fwd = torch.matmul(pts_W, tfs.view(-1, 16)).view(-1, 4, 4).float()
        return T_fwd, pts_W
    
    def save_canonical_weights(self, path):
        points_skinning, face_idx = self.cano_mesh.sample(self.cfg.n_reg_pts * 3, return_index=True)
        points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
        bary_coords = igl.barycentric_coordinates_tri(
            points_skinning,
            self.smpl_verts[self.faces[face_idx, 0], :],
            self.smpl_verts[self.faces[face_idx, 1], :],
            self.smpl_verts[self.faces[face_idx, 2], :],
        )
        vert_ids = self.faces[face_idx, ...]
        pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)
        # adding samples for hand
        points_skinning_hand, face_idx_hand = self.cano_hand_mesh.sample(self.cfg.n_reg_pts , return_index=True)
        points_skinning_hand = points_skinning_hand.view(np.ndarray).astype(np.float32)
        faces_hand = self.cano_hand_mesh.faces
        verts_hand = self.cano_hand_mesh.vertices.view(np.ndarray).astype(np.float32)
        bary_coords_hand = igl.barycentric_coordinates_tri(
            points_skinning_hand,
            verts_hand[faces_hand[face_idx_hand, 0], :],
            verts_hand[faces_hand[face_idx_hand, 1], :],
            verts_hand[faces_hand[face_idx_hand, 2], :],
        )
        vert_ids_hand = faces_hand[face_idx_hand, ...]
        vert_ids_cano = self.hand2cano[vert_ids_hand]
        pts_W_hand = (self.skinning_weights[vert_ids_cano] * bary_coords_hand[..., None]).sum(axis=1)
        points_skinning_hand = torch.from_numpy(points_skinning_hand).cuda()
        pts_W_hand = torch.from_numpy(pts_W_hand).cuda()

        points_skinning = torch.from_numpy(points_skinning).cuda()
        pts_W = torch.from_numpy(pts_W).cuda()

        points_skinning = torch.cat([points_skinning, points_skinning_hand], dim=0)
        # normalize it
        points_skinning = self.cano_aabb.normalize(points_skinning, sym=True)
        pts_W = torch.cat([pts_W, pts_W_hand], dim=0)

        joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
        joint_colors = joint_colors[:,:3]
        points = points_skinning.detach().cpu()
        pred = pts_W.detach().cpu()
        maxjoint_pred_idx = torch.argmax(pred, dim=1)
        pred_colors = joint_colors[maxjoint_pred_idx]
        with open(path, 'w') as f:
            for point, color in zip(points, pred_colors):
                # Write XYZRGB data to the file
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        print("XYZRGB file created successfully.")

        return points_skinning, pts_W

    def sample_skinning_loss(self):
        points_skinning, face_idx = self.cano_mesh.sample(self.cfg.n_reg_pts, return_index=True)
        points_skinning = points_skinning.view(np.ndarray).astype(np.float32)
        bary_coords = igl.barycentric_coordinates_tri(
            points_skinning,
            self.smpl_verts[self.faces[face_idx, 0], :],
            self.smpl_verts[self.faces[face_idx, 1], :],
            self.smpl_verts[self.faces[face_idx, 2], :],
        )
        vert_ids = self.faces[face_idx, ...]
        pts_W = (self.skinning_weights[vert_ids] * bary_coords[..., None]).sum(axis=1)
        # adding samples for hand
        points_skinning_hand, face_idx_hand = self.cano_hand_mesh.sample(self.cfg.n_reg_pts // 4, return_index=True)
        points_skinning_hand = points_skinning_hand.view(np.ndarray).astype(np.float32)
        faces_hand = self.cano_hand_mesh.faces
        verts_hand = self.cano_hand_mesh.vertices.view(np.ndarray).astype(np.float32)
        bary_coords_hand = igl.barycentric_coordinates_tri(
            points_skinning_hand,
            verts_hand[faces_hand[face_idx_hand, 0], :],
            verts_hand[faces_hand[face_idx_hand, 1], :],
            verts_hand[faces_hand[face_idx_hand, 2], :],
        )
        vert_ids_hand = faces_hand[face_idx_hand, ...]
        vert_ids_cano = self.hand2cano[vert_ids_hand]
        pts_W_hand = (self.skinning_weights[vert_ids_cano] * bary_coords_hand[..., None]).sum(axis=1)
        points_skinning_hand = torch.from_numpy(points_skinning_hand).cuda()
        pts_W_hand = torch.from_numpy(pts_W_hand).cuda()

        points_skinning = torch.from_numpy(points_skinning).cuda()
        pts_W = torch.from_numpy(pts_W).cuda()

        points_skinning = torch.cat([points_skinning, points_skinning_hand], dim=0)
        pts_W = torch.cat([pts_W, pts_W_hand], dim=0)

        return points_skinning, pts_W

    def softmax(self, logit):
        # if logit.shape[-1] == 25:
        if logit.shape[-1] == self.d_out:
            w = hierarchical_softmax(logit)
            # w = F.softmax(logit, dim=-1)
        # elif logit.shape[-1] == 24:
        elif logit.shape[-1] != self.d_out:
            w = F.softmax(logit, dim=-1)
        else:
            raise ValueError
        return w

    def get_skinning_loss(self):
        pts_skinning, sampled_weights = self.sample_skinning_loss()
        pts_skinning = self.cano_aabb.normalize(pts_skinning, sym=True) 
        # add some random noise?
        # pts_skinning += torch.randn_like(pts_skinning) * 0.01
        knn_weight = self.query_weights(pts_skinning)

        if self.distill:
            pred_weights = F.grid_sample(self.lbs_voxel_final, pts_skinning.reshape(1, 1, 1, -1, 3), padding_mode='border')
            pred_weights = pred_weights.reshape(24, -1).permute(1, 0)
        else:
            pred_weights = self.lambda_knn_res * knn_weight + (1 - self.lambda_knn_res) * self.softmax(self.lbs_network(pts_skinning))

        skinning_loss = torch.nn.functional.mse_loss(
            pred_weights, sampled_weights, reduction='none').sum(-1).mean()
        # skinning_loss = torch.nn.functional.l1_loss(
        #     pred_weights, sampled_weights, reduction='none').sum(-1).mean()
        # skinning_loss = 20 * torch.nn.functional.mse_loss(
        #     pred_weights, sampled_weights, reduction='mean').mean()
         
        # try entropy loss
        # hand_pred_weights = pred_weights[..., 25:]
        # entropy_loss = 0.1 * torch.nn.functional.cross_entropy(hand_pred_weights, torch.argmax(hand_pred_weights, dim=1)).mean()

        return skinning_loss, pts_skinning, sampled_weights, pred_weights
    
    # get (N, J) representing the joint weights of each point
    def get_xyz_J(self, gaussians):
        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        knn_weights = self.query_weights(xyz_norm)
        pts_W = self.softmax(self.lbs_network(xyz_norm))

        # pts_W = self.lambda_knn_res * knn_weights + (1 - self.lambda_knn_res) * pts_W
        return self.lambda_knn_res * knn_weights + (1 - self.lambda_knn_res) * pts_W



    def forward(self, gaussians, iteration, camera, **kwargs):

        tfs = camera.bone_transforms
        # Gaussian position
        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]

        if iteration < 6000 and iteration %2000 == 0:
            coord_max = np.max(xyz.detach().cpu().numpy(), axis=0)
            coord_min = np.min(xyz.detach().cpu().numpy(), axis=0)
            # hard code the padding as 0.1 here
            # later should be a parameter
            padding_ratio = 0.1
            padding_ratio = np.array(padding_ratio, dtype=np.float32)
            padding = (coord_max - coord_min) * padding_ratio
            coord_max += padding 
            coord_min -= padding

            self.aabb.update(coord_max, coord_min)


        # normalizing position 
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        # get forward transform by weight * bone_transforms for each Gaussian point
        # Joint number is 55 
        # 3 -> 55 
        # N*3 -> N*55 -> N*16
        T_fwd, pts_W = self.get_forward_transform(xyz_norm, tfs)

        deformed_gaussians = gaussians.clone()
        deformed_gaussians.set_fwd_transform(T_fwd.detach())
        deformed_gaussians.set_skinning_weights(pts_W.detach())

        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians, pts_W

    def regularization(self):
        loss_skinning, pts_skinning, sampled_weights, pred_weights = self.get_skinning_loss()
        return {
            'loss_skinning': loss_skinning,
            'pts_skinning': pts_skinning,
            'sampled_weights': sampled_weights,
            'pred_weights': pred_weights
        }


class Smplx_lbs(nn.Module):
    def __init__(self, d_out , cfg):
        
        super(Smplx_lbs, self).__init__()

        self.lbs_network = get_skinning_mlp(3, 25, cfg.skinning_network)
        self.left_hand_network = get_skinning_mlp(3, 15 + 1, cfg.skinning_network)
        self.right_hand_network = get_skinning_mlp(3, 15 + 1, cfg.skinning_network)
        self.d_out = d_out
    
    def forward(self, xyz, iteration = 10):
        pts_W = torch.zeros(xyz.shape[0], self.d_out, device=xyz.device)
        if (iteration < 10):
            pts_W[..., :25] = F.softmax(self.lbs_network(xyz))
        else:
            # with torch.no_grad():
            pts_W[..., :25] = F.softmax(self.lbs_network(xyz), dim = -1)
            left_wrist_W = pts_W[..., 20:21]
            right_wrist_W = pts_W[..., 21:22]
            # import ipdb; ipdb.set_trace()
            left_hand_W = left_wrist_W * F.softmax(self.left_hand_network(xyz), dim = -1)
            right_hand_W = right_wrist_W * F.softmax(self.right_hand_network(xyz), dim = -1)

            pts_W[..., 20] = left_hand_W[..., 0]
            pts_W[..., 21] = right_hand_W[..., 0]
            pts_W[..., 25:] = torch.cat([left_hand_W[..., 1:], right_hand_W[..., 1:]], dim=-1)

        return pts_W

def get_rigid_deform(cfg, metadata):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "smpl_nn": SMPLNN,
        "skinning_field": SkinningField,
        "skinning_field_smplx": SkinningFieldSmplx,
        
    }
    return model_dict[name](cfg, metadata)