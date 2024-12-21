import pickle
import numpy as np
from plyfile import PlyData, PlyElement
import os
import matplotlib.pyplot as plt
import open3d as o3d

def convert_dict_to_ply(dict, ply_path):

    # Example vertices and faces numpy arrays
    vertices = dict['vertices']
    faces = dict['faces']

    # Create a mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Write to a PLY file
    o3d.io.write_triangle_mesh("output.ply", mesh)



if __name__ == '__main__':
    root_dir = "../data_lx/datasets/4Dress/00185/"
    style = "Inner"
    take = "Take1"
    clothe = "Semantic/clothes"
    pkl_file = "cloth-f00011.pkl"
    with open(os.path.join(root_dir, style, take, clothe, pkl_file), 'rb') as f:
        clothes_ply = pickle.load(f)
    mesh_dict = clothes_ply['upper']
    convert_dict_to_ply(mesh_dict, os.path.join(root_dir, "mesh.ply"))