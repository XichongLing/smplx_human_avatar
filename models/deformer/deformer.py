import torch.nn as nn
import torch
from models.deformer.rigid import get_rigid_deform
from models.deformer.non_rigid import get_non_rigid_deform
from models.deformer.garm_simulator import get_garm_simulator
import numpy as np

class Deformer(nn.Module):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.rigid = get_rigid_deform(cfg.rigid, metadata)
        self.non_rigid = get_non_rigid_deform(cfg.non_rigid, metadata)
        # self.garm_simulator = get_garm_simulator(cfg.garm_simulator, metadata)

    def forward(self, gaussians, camera, camera_t, iteration, compute_loss=True):
        loss_reg = {}
        time_enc = time_encoding(camera_t, torch.float32, 4)
        deformed_gaussians, loss_non_rigid = self.non_rigid(gaussians, iteration, camera, compute_loss)
        
        deformed_gaussians, pts_W = self.rigid(deformed_gaussians, iteration, camera)

        gaussians.set_xyz_J(pts_W)
        deformed_gaussians.set_xyz_J(pts_W)


        # get the joint weights of each points
        # gaussians.set_xyz_J(self.rigid.get_xyz_J(deformed_gaussians))
        # if (iteration in [100, 2100, 4100,7100,12000]):
            # import ipdb; ipdb.set_trace()
            # gaussians.save_ply(f"point_cloud/gaussian_{iteration}.ply")
            # gaussians.save_aabb_deformed_ply(f"point_cloud/aabb_deformed_gaussian_{iteration}.txt", self.rigid.aabb)
            # gaussians.save_weights(f"point_cloud/gaussian_weights_{iteration}.txt")
            # save skining weights on canonical mesh
            # self.rigid.save_canonical_weights(f"point_cloud/canonical_weights_{iteration}.txt")
            # deformed_gaussians.save_weights(f"point_cloud/deformed_gaussian_weights_{iteration}.txt")

        loss_reg.update(loss_non_rigid)
        return deformed_gaussians, loss_reg

def get_deformer(cfg, metadata):
    return Deformer(cfg, metadata)

def time_encoding(t, dtype, max_freq=4):
    time_enc = torch.empty(max_freq * 2 + 1, dtype=dtype)

    for i in range(max_freq):
        time_enc[2 * i] = np.sin(2 ** i * torch.pi * t)
        time_enc[2 * i + 1] = np.cos(2 ** i * torch.pi * t)
    time_enc[max_freq * 2] = t
    return time_enc