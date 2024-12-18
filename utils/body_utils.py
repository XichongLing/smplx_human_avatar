import matplotlib.pyplot as plt
import numpy as np
import torch

def ptsw2jointcolor(pts_W):
    model_type = 'smpl' # hard-coded model type, used for skinning weights visualization
    if model_type == 'smplx':
        num_joints = 55 # assuming on smplx model
        joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 20))))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20c')(np.linspace(0, 1, 15))))
        joint_colors = joint_colors[:,:3]
    elif model_type == 'smpl':
        num_joints = 25
        joint_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        joint_colors = np.vstack((joint_colors, plt.cm.get_cmap('tab20b')(np.linspace(0, 1, 5))))
        joint_colors = joint_colors[:,:3]

    joint_indices = torch.argmax(pts_W, dim=1)
    point_colors = joint_colors[joint_indices]
    return point_colors 