import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
import os


SURFACE_LABEL = ['skin', 'hair', 'shoe', 'upper', 'lower', 'outer']

# add ZJUMoCAP dataloader
def get_02v_bone_transforms(Jtr,):
    # Jtr is the joint locations in the SMPL minimal A-pose
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (Jtr.shape[0], 1, 1))

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()  # parent joint location
            t = np.dot(rot, t - t_p)  # relative joint location
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v

def fetchPly(path):
    from scene.gaussian_model import BasicPointCloud
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def pointcloud_offset(xyz_ref, xyz_target):

    mean_ref = (np.min(xyz_ref, axis=0) + np.max(xyz_ref, axis=0)) / 2
    mean_target = (np.min(xyz_target, axis=0) + np.max(xyz_target, axis=0)) / 2
    offset = mean_ref - mean_target
    return offset

def rectify_ply_ply(ply_ref_path, ply_target_path, ply_output_path):
    ply_ref = PlyData.read(open(ply_ref_path, 'rb'))
    ply_target = PlyData.read(open(ply_target_path, 'rb'))
    vertices_ref = np.array(ply_ref['vertices'], )
    vertices_target = np.array(ply_target['vertices'], )
    faces_target = np.array(ply_target['faces'])
    offset = pointcloud_offset(vertices_ref, vertices_target)
    vertices_rectified = vertices_target + offset  
    vertices_rectified = [tuple(vertex) for vertex in vertices_rectified]
    vertices_rectified = np.array(vertices_rectified, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    plydata = PlyData(
        [
            PlyElement.describe(vertices_rectified, 'vertex'),
            faces_target,
        ],
        text=True
    )
    plydata.write(ply_output_path)

def rectify_mesh_ply(mesh_ref, ply_target_path, ply_write_path, ply_output_path):
    ply_target = PlyData.read(open(ply_target_path, 'rb'))
    ply_write = PlyData.read(open(ply_write_path, 'rb'))
    vertices_ref = np.array(mesh_ref.vertices)
    vertices_target = np.asarray(np.array(ply_target['vertex']).tolist())
    vertices_write = np.asarray(np.array(ply_write['vertex']).tolist())
    faces_write = ply_write['face']
    offset = pointcloud_offset(vertices_ref, vertices_target)
    vertices_rectified = vertices_write + offset  
    vertices_rectified = [tuple(vertex) for vertex in vertices_rectified]
    vertices_rectified = np.array(vertices_rectified, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    plydata = PlyData(
        [
            PlyElement.describe(vertices_rectified, 'vertex'),
            faces_write,
        ],
        text=True
    )
    plydata.write(ply_output_path)

class AABB(torch.nn.Module):
    def __init__(self, coord_max, coord_min):
        super().__init__()
        self.register_buffer("coord_max", torch.from_numpy(coord_max).float())
        self.register_buffer("coord_min", torch.from_numpy(coord_min).float())

    def normalize(self, x, sym=False):
        x = (x - self.coord_min) / (self.coord_max - self.coord_min)
        if sym:
            x = 2 * x - 1.
        return x

    def unnormalize(self, x, sym=False):
        if sym:
            x = 0.5 * (x + 1)
        x = x * (self.coord_max - self.coord_min) + self.coord_min
        return x

    def clip(self, x):
        return x.clip(min=self.coord_min, max=self.coord_max)

    def volume_scale(self):
        return self.coord_max - self.coord_min

    def scale(self):
        return math.sqrt((self.volume_scale() ** 2).sum() / 3.)

    def update(self, coord_max, coord_min):
        self.coord_max = torch.from_numpy(coord_max).float().type_as(self.coord_max)
        self.coord_min = torch.from_numpy(coord_min).float().type_as(self.coord_min)

def extract_garm_mesh(dataset_dir, subj, outfit, seq):
    subj_outfit_seq_dir = os.path.join(dataset_dir, subj, outfit, seq)
#     # load basic sequence info
#     basic_info = load_pickle(os.path.join(subj_outfit_seq_dir, 'basic_info.pkl'))
#     scan_frames = basic_info['scan_frames']
#     n_frame = scan_frames[0]
#     # locate scan, label, cloth dir
#     scan_dir = os.path.join(subj_outfit_seq_dir, 'Meshes_pkl')
#     label_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'labels')
#     cloth_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'clothes')
#     os.makedirs(cloth_dir, exist_ok=True)
    
#     # locate save_cloth_fn
#     save_cloth_fn = os.path.join(cloth_dir, 'cloth-f{}.pkl'.format(n_frame))

#     # extract clothes from scan_mesh
#     scan_mesh = load_pickle(os.path.join(scan_dir, 'mesh-f{}.pkl'.format(n_frame)))
#     scan_labels = load_pickle(os.path.join(label_dir, 'label-f{}.pkl'.format(n_frame)))['scan_labels']
#     clothes = extract_label_meshes(scan_mesh['vertices'], scan_mesh['faces'], scan_labels, SURFACE_LABEL, scan_mesh['colors'], scan_mesh['uvs'])
#     save_pickle(save_cloth_fn, clothes)
