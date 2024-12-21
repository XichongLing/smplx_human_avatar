import torch.nn as nn
import torch
from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform
from models.deformer.garm_simulator import get_garm_simulator
import numpy as np
import time
from utils.general_utils import vert2monoply
import os
from utils.general_utils import ptsw2jointcolor
from utils.general_utils import build_rotation, gram_schmidt_batch, to_transform_mat

class Deformer(nn.Module):
    def __init__(self, cfg, metadata, deformer_args):
        super().__init__()
        self.cfg = cfg
        self.vb_mode = cfg.get('vb_mode', 'disable')  
        self.vb_delay = cfg.get('vb_delay', 0)  
        self.trainable_label = deformer_args["trainable_label"]  
        self.save_deform = cfg.get('save_deform', False)
        self.save_video = cfg.get('save_video', False)
        self.dir_save_ply = cfg.get('dir_save_ply', "debug")
        deformer_args.update({'save_deform':self.save_deform, 'save_video':self.save_video, 'dir_save_ply': self.dir_save_ply, 'vb_mode': self.vb_mode, 'vb_delay': self.vb_delay})
        self.rigid = get_rigid_deform(cfg.rigid, metadata, self.vb_mode, self.vb_delay, self.trainable_label)
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata)
        self.garm_simulator = get_garm_simulator(cfg.garm_simulator, metadata, deformer_args)
        if self.save_deform or self.save_video: 
            os.makedirs("assets/garm_debug/{0}".format(self.dir_save_ply), exist_ok=True)
      
    def forward(self, gaussians, camera, camera_t, iteration, compute_loss=True):
        loss_reg = {}
        time_enc = time_encoding(camera_t, torch.float32, 4)
        pure_rigid = True

        # save the canonical gaussians before any deformation
        if self.save_deform:
            if iteration % 2000 == 0 and iteration > 10000 and iteration < 20000:
                pcd_type = "cano"
                garm_color = [0, 1, 1]
                save_ply_layered(gaussians, self.trainable_label, camera, iteration, self.dir_save_ply, pcd_type, garm_color)

        if self.vb_mode == 'disable' or (self.vb_mode == 'two_stage' and iteration <= self.vb_delay):
            if pure_rigid:
                deformed_gaussians = gaussians.clone()
                loss_non_rigid = {}
                if self.non_rigid.feature_dim > 0:
                    setattr(deformed_gaussians, "non_rigid_feature",
                            torch.zeros(gaussians.get_xyz.shape[0], self.non_rigid.feature_dim).cuda())
            else:
                deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)
            deformed_gaussians, pts_W = self.rigid(deformed_gaussians, iteration, camera)
            if self.vb_mode == 'two_stage' and iteration == self.vb_delay:
                # extract the virtual bones from the existing garments gaussians
                virtual_joints = self.garm_simulator.extract_virtual_bones(gaussians)
        elif self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            deformed_gaussians = gaussians.clone()
            loss_non_rigid = {}
            if self.non_rigid.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature",
                        torch.zeros(gaussians.get_xyz.shape[0], self.non_rigid.feature_dim).cuda())
            deformed_gaussians.init_fwd_transform(camera.transl, camera.root_orient_mat)
            deformed_gaussians, pts_W = self.rigid(deformed_gaussians, iteration, camera)
            deformed_gaussians, nodes_deformed = self.garm_simulator(deformed_gaussians, iteration, camera, time_enc)
            nodes_d_smpl = self.rigid.get_garm_deformation(self.garm_simulator.vb_model.get_virtual_joints(), camera)
            tf_reg_loss = get_tf_reg_loss(nodes_deformed, nodes_d_smpl)
            loss_reg.update({"tf_reg_loss": tf_reg_loss})
        else:
            raise ValueError("Invalid vb_mode")

        # save the canonical gaussians after all deformations
        if self.save_deform:
            if iteration % 2000 == 0 and iteration > 10000 and iteration < 20000:
                pcd_type = "deformed"
                garm_color = [0, 1, 1]
                save_ply_layered(deformed_gaussians, self.trainable_label, camera, iteration, self.dir_save_ply, pcd_type, garm_color)
        

        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg, None, None


    def get_vb_deformation_regularization(self,):
        xyz, T_fwd_gs = self.garm_simulator.sample_garm_points()
        T_fwd_rigid = self.rigid.get_fwd_transform(xyz)

    def update_learning_rate_vb(self, iteration):
        if self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            self.garm_simulator.vb_model.update_learning_rate(iteration)
    def optimize_vb(self, iteration):
        if self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            self.garm_simulator.vb_model.optimize()

def get_deformer(cfg, metadata, trainable_label):
    return Deformer(cfg, metadata, trainable_label)

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

def save_ply_layered(gaussians, trainable_label, camera, iteration, dir_save_ply, pcd_type, garm_color):
    garm_xyz = gaussians.get_xyz_by_category(1, trainable_label)
    body_xyz = gaussians.get_xyz_by_category(0, trainable_label)
    # garm_xyz = garm_xyz - torch.tensor(camera.transl).cuda()
    # body_xyz = body_xyz - torch.tensor(camera.transl).cuda()
    vert2monoply(garm_xyz, "assets/garm_debug/{0}/{2}_garm_{1}.ply".format(dir_save_ply, iteration, pcd_type), garm_color)
    vert2monoply(body_xyz, "assets/garm_debug/{0}/{2}_body_{1}.ply".format(dir_save_ply, iteration, pcd_type))
