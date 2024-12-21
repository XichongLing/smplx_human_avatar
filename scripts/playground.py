from utils.general_utils import vert2monoply
import torch

if __name__ == '__main__':
    xyz = torch.rand(1000, 3)
    file_name = "test.ply"  
    vert2monoply(xyz, file_name)