import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d.ops as ops
import trimesh
import igl

from utils.general_utils import build_rotation
from models.network_utils import get_skinning_mlp, get_ImplicitNet, get_deformation_mlp
from utils.dataset_utils import AABB 

import copy

import matplotlib.pyplot as plt
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


# need to change here
class DeformationGraph(Garm_Simulator):
    def __init__(self, cfg, metadata):
        super().__init__(cfg)
        self.virtual_bones = metadata["virtual_bones"]
        self.virtual_joints = self.virtual_bones.vertices
        self.num_vb = self.virtual_joints.shape[0]
        self.aabb = metadata["aabb"]
        self.root_orient = metadata["root_orient"]
        self.trans = metadata["trans"]

        self.distill = cfg.distill
        d, h, w = cfg.res // cfg.z_ratio, cfg.res, cfg.res
        self.resolution = (d, h, w)
        if self.distill:
            self.grid = create_voxel_grid(d, h, w).cuda()

        # self.deformation_graph = get_ImplicitNet(cfg)
        # input_dim = 3 + dim(time_enc)
        # output_dim = 9 (rotation) + 3 (translation)
        self.deformation_graph = get_deformation_mlp(cfg, 3, 12, self.num_vb, cfg.deformation_network)
        # self.lbs_network = Smplx_lbs(d_out= 55, cfg=cfg)
        # need to further check d_out is the num_joint?
        # wrist part: 21 22
        # - theoretically yes, lbs_network input: sampled cano points, output: weights for each joint
        self.d_out = cfg.d_out



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

    # get (N, J) representing the joint weights of each point
    def get_xyz_J(self, gaussians):
        xyz = gaussians.get_xyz
        xyz_norm = self.aabb.normalize(xyz, sym=True)
        knn_weights = self.query_weights(xyz_norm)
        pts_W = self.softmax(self.lbs_network(xyz_norm))

        # pts_W = self.lambda_knn_res * knn_weights + (1 - self.lambda_knn_res) * pts_W
        return self.lambda_knn_res * knn_weights + (1 - self.lambda_knn_res) * pts_W


    def forward(self, gaussians, iteration, time_enc, **kwargs):

        # Gaussian position
        xyz = gaussians.get_xyz
        n_pts = xyz.shape[0]
        cond = 'smpl' # or smpl and time?
        virtual_weights = self.weighted_knn(xyz) # size of (N, num_vb)
        vb_rotation, vb_translation = self.get_vb_deformation(self.virtual_joints, cond, time_enc)
        weighted_rotation = torch.matmul(virtual_weights, vb_rotation).squeeze(1)
        weighted_translation = torch.matmul(virtual_weights, vb_translation).squeeze(1)
        
        deformed_gaussians = gaussians.clone()

        #Todo: produce the transformed xyz
        homo_coord = torch.ones(n_pts, 1, dtype=torch.float32, device=xyz.device)
        x_hat_homo = torch.cat([xyz, homo_coord], dim=-1).view(n_pts, 4, 1)
        x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
        deformed_gaussians._xyz = x_bar

        rotation_hat = build_rotation(gaussians._rotation)
        rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
        setattr(deformed_gaussians, 'rotation_precomp', rotation_bar)
        # deformed_gaussians._rotation = tf.matrix_to_quaternion(rotation_bar)
        # deformed_gaussians._rotation = rotation_matrix_to_quaternion(rotation_bar)

        return deformed_gaussians
    
    def query_weights(self, xyz):
        # find the nearest vertex
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.virtual_joints.unsqueeze(0))
        p_idx = knn_ret.idx.squeeze()
        pts_W = self.skinning_weights_tensor[p_idx, :]
        #Todo: decide a correct skinning weight
        return pts_W

    def weighted_knn(self, xyz):
        knn_ret = ops.knn_points(xyz.unsqueeze(0), self.virtual_joints.unsqueeze(0), K=4)
        p_idx = knn_ret.idx.squeeze()
        p_dists = knn_ret.dists.squeeze()
        # weighted_knn = return an array of N * xyz[0], the distance of the N virtual joints with respect to the i-th xyz, only the
        # nearest k positions set to non-zero, the valid k position should be normalized
        weighted_knn = torch.zeros(xyz.shape[0], self.num_vb, device=xyz.device)
        weighted_knn.scatter_(1, p_idx, p_dists)
        row_sum = torch.sum(weighted_knn, dim=1)
        weighted_knn = weighted_knn / row_sum
        return weighted_knn
    
    # concatenate the nodes and time_embedding, producing the rotation and translation for all virtual bones
    def get_vb_deformation(self, nodes, cond, time_enc):
        transform_mat = self.deformation_graph(nodes, cond, time_enc).unsqueeze(0)      # 6 DOF
        
        # Version 2
        # smpl_root_orient_mat = batch_rodrigues(self.root_orient)
        # smpl_root_orient_mat = to_transform_mat(smpl_root_orient_mat, torch.zeros([smpl_root_orient_mat.shape[0], 3, 1]).cuda()).unsqueeze(0).detach()

        # transform_mat = torch.matmul(smpl_root_orient_mat.expand(-1, transform_mat.shape[1], -1, -1), transform_mat)
        # transform_mat[:, :, :3, 3] = transform_mat[:, :, :3, 3] + self.trans.unsqueeze(1)
        
        # skinning_weights_self = torch.eye(nodes.shape[0]).cuda()
        # nodes_deformed = skinning(nodes[None], skinning_weights_self[None], transform_mat, inverse=False, return_T=False)
        # return nodes_deformed.squeeze(0)  
        # return transform_mat
        return transform_mat[:,:3], transform_mat[:,3:]


    def get_forward_transform(self, nodes, cond, time_enc):
        transformation = self.deformation_graph(nodes, cond, time_enc) # predict the 6 DoF
        rot = transformation[:, :, :3]
        trans = transformation[0, :, 3:].unsqueeze(-1)
        rot_mat = batch_rodrigues(rot[0]) # from axis-angle to rotation matrix
        transform_mat = to_transform_mat(rot_mat, trans)
        return transform_mat
    


    def regularization(self):
        loss_skinning, pts_skinning, sampled_weights, pred_weights = self.get_skinning_loss()
        return {
            'loss_skinning': loss_skinning,
            'pts_skinning': pts_skinning,
            'sampled_weights': sampled_weights,
            'pred_weights': pred_weights
        }



def get_garm_simulator(cfg, metadata):
    name = cfg.name
    model_dict = {
        "identity": Identity,
        "deformation_graph": DeformationGraph,
        
    }
    return model_dict[name](cfg, metadata)

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
    return rot_mat

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