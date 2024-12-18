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
        
        # save the garments gaussians in canonical space
        # if self.save_deform:
            # if (iteration % 500 == 0 and iteration > 8000 and iteration < 15000 and self.save_deform) or (iteration in [1000,2000, 3000, 4000, 5000,6000, 7000]):
            #     garm_xyz = gaussians.get_xyz_by_category(1, self.trainable_label)   
            #     garm_color = [0, 1, 1]
            #     vert2monoply(garm_xyz, "assets/garm_debug/{0}/cano_garm_{1}.ply".format(self.dir_save_ply, iteration), garm_color)

            # if iteration in [3000,5000, 6000, 7500, 10000,12000]:
            #     body_xyz = gaussians.get_xyz_by_category(0, self.trainable_label)
            #     vert2monoply(body_xyz, "assets/garm_debug/{0}/cano_body_{1}.ply".format(self.dir_save_ply, iteration))

            # if iteration in [7000, 7500, 8000, 8500, 10000, 15000]:
            #     vb_color = [1, 0, 0.5]
            #     vert2monoply(self.garm_simulator.vb_model.get_virtual_joints(), "assets/garm_debug/{0}/virtual_joints_{1}.ply".format(self.dir_save_ply, iteration), vb_color)

        if self.save_deform:
            if iteration % 1000 == 0 and iteration > 8000 and iteration < 15000:
                pcd_type = "cano"
                garm_color = [0, 1, 1]
                save_ply_layered(gaussians, self.trainable_label, camera, iteration, self.dir_save_ply, pcd_type, garm_color)
        if self.vb_mode == 'disable' or (self.vb_mode == 'two_stage' and iteration <= self.vb_delay):
            deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)
            # deformed_gaussians = gaussians.clone()
            # loss_non_rigid = {}
            # if self.non_rigid.feature_dim > 0:
            #     setattr(deformed_gaussians, "non_rigid_feature",
            #             torch.zeros(gaussians.get_xyz.shape[0], self.non_rigid.feature_dim).cuda())
        elif self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            deformed_gaussians = gaussians.clone()
            loss_non_rigid = {}
            if self.non_rigid.feature_dim > 0:
                setattr(deformed_gaussians, "non_rigid_feature",
                        torch.zeros(gaussians.get_xyz.shape[0], self.non_rigid.feature_dim).cuda())
            # deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)

        # save the non-rigid gaussians _xyz for deubgging
        # non_rigid_xyz = deformed_gaussians.get_xyz
        # non_rigid_xyz = gaussians.get_xyz
        # smpl_root_orient = camera.root_orient_mat.cuda()
        # smpl_root_orient_mat = smpl_root_orient.unsqueeze(0)
        # non_rigid_xyz = torch.matmul(smpl_root_orient_mat.expand(non_rigid_xyz.shape[0], -1, -1), non_rigid_xyz.unsqueeze(2)).squeeze(2)
        # non_rigid_xyz = non_rigid_xyz + torch.tensor(camera.transl).cuda()

        if self.vb_mode == 'two_stage' and iteration == self.vb_delay:
            # extract the virtual bones from the existing garments gaussians
            virtual_joints = self.garm_simulator.extract_virtual_bones(gaussians)
            if self.save_deform:
                vert2monoply(virtual_joints, "assets/garm_debug/{0}/virtual_joints.ply".format(self.dir_save_ply), [1,0,0])
        
        if self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            # deformed_gaussians.init_fwd_transform(camera.transl, camera.root_orient_mat)
            pass
        elif self.vb_mode == 'disable' or (self.vb_mode == 'two_stage' and iteration <= self.vb_delay):
            pass
        else:
            raise ValueError("Invalid vb_mode")


        # if iteration % 100 == 0:
        #     print("itertaion", iteration, "before rigid")
        #     print("number of body gaussians:", (gaussians._label==0.).sum(dim=0)[0])
        #     print("number of garments gaussians:", (gaussians._label==1.).sum(dim=0)[0])

        deformed_gaussians, pts_W = self.rigid(deformed_gaussians, iteration, camera)
        visualize_joints = False
        if visualize_joints:
            joint_colors = ptsw2jointcolor(pts_W)
            joint_colors = torch.tensor(joint_colors).cuda()
        else:
            joint_colors = None

        if self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            deformed_gaussians, nodes_deformed = self.garm_simulator(deformed_gaussians, iteration, camera, time_enc)
            virtual_joints = self.garm_simulator.get_virtual_joints()
            nodes_d_smpl = self.rigid.get_garm_deformation(virtual_joints, camera)
            tf_reg_loss = get_tf_reg_loss(nodes_deformed, nodes_d_smpl)
            loss_reg.update({"tf_reg_loss": tf_reg_loss})
            garm_color = [0, 1, 1]
            if iteration % 500 == 0 and iteration > 8000 and iteration < 15000 and self.save_deform:
                pcd_type = "deformed"
                save_ply_layered(deformed_gaussians, self.trainable_label, camera, iteration, self.dir_save_ply, pcd_type, garm_color)
        elif self.vb_mode == 'disable' or (self.vb_mode == 'two_stage' and iteration <= self.vb_delay):
            pass
        else:   
            raise ValueError("Invalid vb_mode")

        

        loss_reg.update(loss_non_rigid)
        non_rigid_xyz = {}
        return deformed_gaussians, loss_reg, joint_colors, non_rigid_xyz


    def get_vb_deformation_regularization(self,):
        xyz, T_fwd_gs = self.garm_simulator.sample_garm_points()
        T_fwd_rigid = self.rigid.get_fwd_transform(xyz)

    def update_learning_rate_vb(self, iteration):
        if self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            self.garm_simulator.vb_model.update_learning_rate(iteration)

    def optimize_vb(self, iteration):
        if self.vb_mode == 'enable' or (self.vb_mode == 'two_stage' and iteration > self.vb_delay):
            self.garm_simulator.vb_model.optimize()

def get_deformer(cfg, metadata, deformer_args):
    return Deformer(cfg, metadata, deformer_args)

def time_encoding(t, dtype, max_freq=4):
    time_enc = torch.empty(max_freq * 2 + 1, dtype=dtype)

    for i in range(max_freq):
        time_enc[2 * i] = np.sin(2 ** i * torch.pi * t)
        time_enc[2 * i + 1] = np.cos(2 ** i * torch.pi * t)
    time_enc[max_freq * 2] = t
    return time_enc

def get_tf_reg_loss(nodes_d_garm, nodes_d_smpl):
    if nodes_d_garm is not None and nodes_d_smpl is not None:
        l2_loss = nn.MSELoss(reduction='mean')
        return l2_loss(nodes_d_smpl, nodes_d_garm)
    else:
        return None
    
def save_ply_layered(gaussians, trainable_label, camera, iteration, dir_save_ply, pcd_type, garm_color):
    garm_xyz = gaussians.get_xyz_by_category(1, trainable_label)
    body_xyz = gaussians.get_xyz_by_category(0, trainable_label)
    garm_xyz = garm_xyz - torch.tensor(camera.transl).cuda()
    body_xyz = body_xyz - torch.tensor(camera.transl).cuda()
    vert2monoply(garm_xyz, "assets/garm_debug/{0}/{2}_garm_{1}.ply".format(dir_save_ply, iteration, pcd_type), garm_color)
    vert2monoply(body_xyz, "assets/garm_debug/{0}/{2}_body_{1}.ply".format(dir_save_ply, iteration, pcd_type))
