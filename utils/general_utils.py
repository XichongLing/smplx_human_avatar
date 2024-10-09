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
import os
import sys
from datetime import datetime
import numpy as np
import random
import torchvision
import tqdm
import torch.nn as nn
import lpips
import cv2
from skimage.metrics import structural_similarity as compute_ssim

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid
from torch import Tensor
import open3d as o3d

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) format.

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(*, 3, 3)`.
        eps: small value to avoid zero division.

    Return:
        the rotation in quaternion with shape :math:`(*, 4)`.

    Example:
        >>> input = tensor([[1., 0., 0.],
        ...                       [0., 1., 0.],
        ...                       [0., 0., 1.]])
        >>> rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps)
        tensor([1., 0., 0., 0.])
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond() -> torch.Tensor:
        sq = torch.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3() -> torch.Tensor:
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def quaternion_multiply(r, s):
    r0, r1, r2, r3 = r.unbind(-1)
    s0, s1, s2, s3 = s.unbind(-1)
    t0 = r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3
    t1 = r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2
    t2 = r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1
    t3 = r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0
    t = torch.stack([t0, t1, t2, t3], dim=-1)
    return t

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    if r.shape[-1] == 4:
        # quaternion to matrix
        R = build_rotation(r)
    else:
        R = r

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def fix_random(seed):
    if seed >= 0:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

# evaluation metrics
class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.lpips = LPIPS()

    def forward(self, inputs, targets):
        psnr = self.psnr(inputs, targets)
        ssim = self.ssim(inputs, targets)
        lpips_ = self.lpips(inputs, targets)
        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips_,
        }

class PSNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets) ** 2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        device = inputs.device
        inputs = inputs.cpu().numpy()
        targets = targets.cpu().numpy()
        if valid_mask is not None:
            valid_mask = valid_mask.cpu().numpy()
            x, y, w, h = cv2.boundingRect(valid_mask.astype(np.uint8))
            img_pred = inputs[y:y + h, x:x + w]
            img_gt = targets[y:y + h, x:x + w]
        else:
            img_pred = inputs
            img_gt = targets

        # compute ssim
        ssim = compute_ssim(img_pred, img_gt, channel_axis=0)
        ssim = torch.tensor(ssim, device=device)
        return ssim


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
        self.loss_fn_vgg.eval()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        if valid_mask is not None:
            x, y, w, h = cv2.boundingRect(valid_mask.cpu().numpy().astype(np.uint8))
            img_pred = inputs[:, y:y + h, x:x + w]
            img_gt = targets[:, y:y + h, x:x + w]
        else:
            img_pred = inputs
            img_gt = targets

        score = self.loss_fn_vgg(img_pred, img_gt, normalize=True)
        return score.flatten()

class PSEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.cuda()
        self.eval()

    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        rgb = rgb.unsqueeze(0)
        rgb_gt = rgb_gt.unsqueeze(0)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }
    
def npz_to_video_grid(data_path, out_path, num_frames=None, fps=8, num_videos=None, nrow=None, verbose=True):
    if isinstance(data_path, str):
        videos = load_num_videos(data_path, num_videos)
    elif isinstance(data_path, np.ndarray):
        videos = data_path
    else:
        raise Exception
    n, t, h, w, c = videos.shape

    videos_th = []
    for i in range(n):
        video = videos[i, :, :, :, :]
        images = [video[j, :, :, :] for j in range(t)]
        images = [to_tensor(img) for img in images]
        video = torch.stack(images)
        videos_th.append(video)

    if num_frames is None:
        num_frames = videos.shape[1]
    if verbose:
        videos = [fill_with_black_squares(v, num_frames) for v in tqdm(videos_th, desc='Adding empty frames')]  # NTCHW
    else:
        videos = [fill_with_black_squares(v, num_frames) for v in videos_th]  # NTCHW

    frame_grids = torch.stack(videos).permute(1, 0, 2, 3, 4)  # [T, N, C, H, W]
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(n)))
    if verbose:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in tqdm(frame_grids, desc='Making grids')]
    else:
        frame_grids = [make_grid(fs, nrow=nrow) for fs in frame_grids]

    if os.path.dirname(out_path) != "":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frame_grids = (torch.stack(frame_grids) * 255).to(torch.uint8).permute(0, 2, 3, 1)  # [T, H, W, C]
    torchvision.io.write_video(out_path, frame_grids, fps=fps, video_codec='h264', options={'crf': '10'})

def load_num_videos(data_path, num_videos):
    # data_path can be either data_path of np array
    if isinstance(data_path, str):
        videos = np.load(data_path)['arr_0']  # NTHWC
    elif isinstance(data_path, np.ndarray):
        videos = data_path
    else:
        raise Exception

    if num_videos is not None:
        videos = videos[:num_videos, :, :, :, :]
    return videos

def fill_with_black_squares(video, desired_len: int) -> Tensor:
    if len(video) >= desired_len:
        return video

    return torch.cat([
        video,
        torch.zeros_like(video[0]).unsqueeze(0).repeat(desired_len - len(video), 1, 1, 1),
    ], dim=0)

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu

def gram_schmidt_batch(R_list):
    # R_list is a list of N rotation matrix of shape (N, 3, 3)
    R_orthonormal = torch.zeros_like(R_list)
    for i in range(R_list.shape[0]):
        R_orthonormal[i] = gram_schmidt(R_list[i])
    return R_orthonormal

def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
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

def euler2rotmat(euler):
    # the similar function in pytorch3d.transforms has significant math error compared to this one
    # don't know why
    sx = torch.sin(euler[:, 0]).view((-1, 1))
    sy = torch.sin(euler[:, 1]).view((-1, 1))
    sz = torch.sin(euler[:, 2]).view((-1, 1))
    cx = torch.cos(euler[:, 0]).view((-1, 1))
    cy = torch.cos(euler[:, 1]).view((-1, 1))
    cz = torch.cos(euler[:, 2]).view((-1, 1))

    mat_flat = torch.hstack([cy * cz,
                             sx * sy * cz - sz * cx,
                             sy * cx * cz + sx * sz,
                             sz * cy,
                             sx * sy * sz + cx * cz,
                             sy * sz * cx - sx * cz,
                             -sy,
                             sx * cy,
                             cx * cy])
    return mat_flat.view((-1, 3, 3))

def vert2monoply(xyz, ply_file_name):
    xyz = xyz.detach().cpu()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(ply_file_name, pcd)
