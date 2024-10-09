import torch.nn as nn
import torch
from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform
from models.deformer.garm_simulator import get_garm_simulator
import numpy as np
import time
from utils.general_utils import vert2monoply

class Deformer(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.vb_mode = cfg.get('vb_mode', 'disable')  
        self.vb_delay = cfg.get('vb_delay', 0)  
        self.rigid = get_rigid_deform(cfg.rigid, metadata, self.vb_mode, self.vb_delay)
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata)
        self.garm_simulator = get_garm_simulator(cfg.garm_simulator, metadata, self.vb_mode, self.vb_delay)

    def forward(self, gaussians, camera, camera_t, iteration, compute_loss=True):
        loss_reg = {}
        time_enc = time_encoding(camera_t, torch.float32, 4)

        deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)

        if self.vb_mode == 'two_stage' and iteration == self.vb_delay:
            # extract the virtual bones from the existing garments gaussians
            virtual_joints = self.garm_simulator.extract_virtual_bones(gaussians)

            garm_xyz = gaussians.get_xyz_by_category(1)
            body_xyz = gaussians.get_xyz_by_category(0)
            vert2monoply(garm_xyz, "assets/garm_gaussians.ply")
            vert2monoply(body_xyz, "assets/body_gaussians.ply") 
            vert2monoply(virtual_joints, "assets/virtual_joints.ply")
        
        if self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            deformed_gaussians.init_fwd_transform(camera.transl, camera.root_orient_mat)
        elif self.vb_mode == 'disable' or (self.vb_mode == 'two_stage' and iteration <= self.vb_delay):
            pass
        else:
            raise ValueError("Invalid vb_mode")


        # if iteration % 100 == 0:
        #     print("itertaion", iteration, "before rigid")
        #     print("number of body gaussians:", (gaussians._label==0.).sum(dim=0)[0])
        #     print("number of garments gaussians:", (gaussians._label==1.).sum(dim=0)[0])

        deformed_gaussians, pts_W = self.rigid(deformed_gaussians, iteration, camera)


        if self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            deformed_gaussians, nodes_deformed = self.garm_simulator(deformed_gaussians, iteration, camera, time_enc)
            virtual_joints = self.garm_simulator.get_virtual_joints()
            nodes_d_smpl = self.rigid.get_garm_deformation(virtual_joints, camera)
            tf_reg_loss = get_tf_reg_loss(nodes_deformed, nodes_d_smpl)
            loss_reg.update({"tf_reg_loss": tf_reg_loss})
        elif self.vb_mode == 'disable' or (self.vb_mode == 'two_stage' and iteration <= self.vb_delay):
            pass
        else:   
            raise ValueError("Invalid vb_mode")

        

        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg


    def get_vb_deformation_regularization(self,):
        xyz, T_fwd_gs = self.garm_simulator.sample_garm_points()
        T_fwd_rigid = self.rigid.get_fwd_transform(xyz)

def get_deformer(cfg, metadata):
    return Deformer(cfg, metadata)

def time_encoding(t, dtype, max_freq=4):
    time_enc = torch.empty(max_freq * 2 + 1, dtype=dtype)

    for i in range(max_freq):
        time_enc[2 * i] = np.sin(2 ** i * torch.pi * t)
        time_enc[2 * i + 1] = np.cos(2 ** i * torch.pi * t)
    time_enc[max_freq * 2] = t
    return time_enc

def get_tf_reg_loss(nodes_d_garm, nodes_d_smpl):
    l2_loss = nn.MSELoss(reduction='mean')
    return l2_loss(nodes_d_smpl, nodes_d_garm)
