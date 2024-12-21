import torch
import os
import numpy as np
import glob
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel
from preprocess_datasets.easymocap.smplmodel import load_model
import argparse
import pytorch3d.ops as ops
import open3d as o3d

""" 
This script is used to generate human body cloudpoints from the minimal shape data under different bone transformations
"""

def vert_deformation(verts, bone_transforms, skinning_weights):
    '''
    deform the canonical vertices based on the bone_transforms,
    return the deformed vertices
    '''

    n_verts = verts.shape[0]  
    pts_W = skinning_weights
    T_fwd = torch.matmul(pts_W, bone_transforms.view(-1, 16)).view(n_verts, 4, 4).float()

    homo_coord = torch.ones(n_verts, 1, dtype=torch.float32, device=verts.device)
    x_hat_homo = torch.cat([verts, homo_coord], dim=-1).view(n_verts, 4, 1)
    x_bar = torch.matmul(T_fwd, x_hat_homo)[:, :3, 0]
    return x_bar

def get_frame_data(data_path):
    model_dict = np.load(data_path, allow_pickle=True)
    minimal_shape = model_dict['minimal_shape']
    bone_transforms = model_dict['bone_transforms'].astype(np.float32)
    trans = model_dict['transl'].astype(np.float32)
    root_orient = model_dict['global_orient'].astype(np.float32)
    if minimal_shape.dtype == np.float16:
        minimal_shape = minimal_shape.astype(np.float32)
        minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
    else:
        minimal_shape = minimal_shape.astype(np.float32)
    return minimal_shape, bone_transforms, trans

def get_smpl_data(gender):
    faces = np.load('body_models/misc/faces.npz')['faces']
    skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
    posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
    J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))

    v_templates = np.load('body_models/misc/v_templates.npz')
    shapedirs = dict(np.load('body_models/misc/shapedirs_all.npz'))
    for k in list(shapedirs.keys()):
        shapedirs[k] = shapedirs[k][:, :, :10]
    kintree_table = np.load('body_models/misc/kintree_table.npy')
    return skinning_weights[gender].astype(np.float32)  

def frame_pose_generation(file_path, gender):
    if file_path != None:
        cano_verts, bone_transforms, trans = get_frame_data(file_path)
        bone_transforms[:, :3, 3] += trans.reshape(1, 3)  # add global offset
        cano_verts = cano_verts.astype(np.float32)
        skinning_weights = get_smpl_data(gender)
        cano_verts = torch.tensor(cano_verts)
        bone_transforms = torch.tensor(bone_transforms)
        skinning_weights = torch.tensor(skinning_weights)
        deformed_verts = vert_deformation(cano_verts, bone_transforms, skinning_weights)
    return deformed_verts   

if __name__ == '__main__':
    root_dir = '../data_lx/datasets/4Dress'  
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', default='00187', help='subj name')
    parser.add_argument('--outfit', default='Inner', help='outfit name')
    parser.add_argument('--seq', default='Take6', help='seq name list')
    gender = 'female'
    args = parser.parse_args()
    model_files = sorted(glob.glob(os.path.join(root_dir, args.subj, args.outfit, args.seq, 'SMPL_processed/*.npz')))
    for file in model_files:
        deformed_verts = frame_pose_generation(file, gender)
        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(deformed_verts.cpu().numpy())
        output_file = file.replace('.npz', '.ply')
        # save_path = os.path.join(root_dir, args.subj, args.outfit, args.seq, 'SMPL_processed', output_file)
        save_path = output_file
        o3d.io.write_point_cloud(save_path, pcd)
