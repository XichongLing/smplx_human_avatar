import pickle
import numpy as np
from plyfile import PlyData, PlyElement
import os
import matplotlib.pyplot as plt
import open3d as o3d
# from pickle import load_pickle, save_pickle
import trimesh
import torch

SURFACE_LABEL = ['skin', 'hair', 'shoe', 'upper', 'lower', 'outer']

# generate the mesh files for different garments on the specified sequence
def extract_garm_mesh(dataset_dir, subj, outfit, seq):
    subj_outfit_seq_dir = os.path.join(dataset_dir, subj, outfit, seq)
    # load basic sequence info
    basic_info = load_pickle(os.path.join(subj_outfit_seq_dir, 'basic_info.pkl'))
    scan_frames = basic_info['scan_frames']
    # for 4d dress, the first frame is normally A pose
    n_frame = scan_frames[0]

    scan_dir = os.path.join(subj_outfit_seq_dir, 'Meshes_pkl')
    label_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'labels')
    cloth_dir = os.path.join(subj_outfit_seq_dir, 'Semantic', 'clothes')
    os.makedirs(cloth_dir, exist_ok=True)
    
    # locate save_cloth_fn
    save_cloth_fn = os.path.join(cloth_dir, 'cloth-f{}.pkl'.format(n_frame))

    # extract clothes from scan_mesh
    scan_mesh = load_pickle(os.path.join(scan_dir, 'mesh-f{}.pkl'.format(n_frame)))
    scan_labels = load_pickle(os.path.join(label_dir, 'label-f{}.pkl'.format(n_frame)))['scan_labels']
    clothes = extract_label_meshes(scan_mesh['vertices'], scan_mesh['faces'], scan_labels, SURFACE_LABEL, scan_mesh['colors'], scan_mesh['uvs'])
    save_pickle(save_cloth_fn, clothes)

    # extract label meshes from scan_mesh
def extract_label_meshes(vertices, faces, labels, surface_labels, colors=None, uvs=None):
    # init label_meshes and face_labels
    label_meshes = dict()
    face_labels = labels[faces]
    # loop over all labels
    for nl in range(len(surface_labels)):
        # skip empty label
        if np.sum(labels == nl) == 0: continue
        # find label faces: with label vertices == 3
        vertex_label_nl = np.where(labels == nl)[0]
        face_label_nl = np.where(np.sum(face_labels == nl, axis=-1) == 3)[0]
        # find correct indices
        correct_indices = (np.zeros(labels.shape[0]) - 1).astype(int)
        correct_indices[vertex_label_nl] = np.arange(vertex_label_nl.shape[0])
        # extract label_mesh[vertices, faces]
        label_meshes[surface_labels[nl]] = {'vertices': vertices[vertex_label_nl], 'faces': correct_indices[faces[face_label_nl]]}
        # extract label_mesh colors
        label_meshes[surface_labels[nl]]['colors'] = colors[vertex_label_nl] if colors is not None else None
        # extract label_mesh uvs
        label_meshes[surface_labels[nl]]['uvs'] = uvs[vertex_label_nl] if uvs is not None else None
    return label_meshes

def convert_dict_to_ply(dict, ply_path):

    # Example vertices and faces numpy arrays
    vertices = dict['vertices']
    faces = dict['faces']

    # Create a mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Write to a PLY file
    o3d.io.write_triangle_mesh(ply_path, mesh)

def get_ref_smpl_offset(cano_smpl_path, garm_dict, target_body_dict):
    cano_smpl = trimesh.load(cano_smpl_path)
    cano_smpl_x_max = torch.max(torch.tensor(cano_smpl.vertices)[:,0])
    cano_smpl_x_min = torch.min(torch.tensor(cano_smpl.vertices)[:,0])
    cano_smpl_y_max = torch.max(torch.tensor(cano_smpl.vertices)[:,1])
    cano_smpl_y_min = torch.min(torch.tensor(cano_smpl.vertices)[:,1])
    cano_smpl_z_max = torch.max(torch.tensor(cano_smpl.vertices)[:,2])
    cano_smpl_z_min = torch.min(torch.tensor(cano_smpl.vertices)[:,2])
    cano_x_gap = cano_smpl_x_max - cano_smpl_x_min
    cano_y_gap = cano_smpl_y_max - cano_smpl_y_min
    cano_z_gap = cano_smpl_z_max - cano_smpl_z_min
    print("cano_x_gap: ", cano_x_gap)
    print("cano_y_gap: ", cano_y_gap)
    print("cano_z_gap: ", cano_z_gap)
    vertices = target_body_dict['vertices']
    # normalize the vertices
    target_x_max = torch.max(torch.tensor(vertices[:,0]))
    target_x_min = torch.min(torch.tensor(vertices[:,0]))
    target_y_max = torch.max(torch.tensor(vertices[:,1]))
    target_y_min = torch.min(torch.tensor(vertices[:,1]))
    target_z_max = torch.max(torch.tensor(vertices[:,2]))
    target_z_min = torch.min(torch.tensor(vertices[:,2]))
    target_x_gap = target_x_max - target_x_min
    target_y_gap = target_y_max - target_y_min
    target_z_gap = target_z_max - target_z_min
    normalized_vertices = (torch.tensor(vertices) - torch.tensor([target_x_min, target_y_min, target_z_min])) / torch.tensor([target_x_gap, target_y_gap, target_z_gap])
    target_x_max = torch.max(torch.tensor(normalized_vertices[:,0]))
    target_x_min = torch.min(torch.tensor(normalized_vertices[:,0]))
    target_y_max = torch.max(torch.tensor(normalized_vertices[:,1]))
    target_y_min = torch.min(torch.tensor(normalized_vertices[:,1]))
    target_z_max = torch.max(torch.tensor(normalized_vertices[:,2]))
    target_z_min = torch.min(torch.tensor(normalized_vertices[:,2]))
    target_x_gap = target_x_max - target_x_min
    target_y_gap = target_y_max - target_y_min
    target_z_gap = target_z_max - target_z_min

    print("target_x_gap: ", target_x_gap)
    print("target_y_gap: ", target_y_gap)
    print("target_z_gap: ", target_z_gap)
    import ipdb; ipdb.set_trace()


    import ipdb; ipdb.set_trace() 


# load data from pkl_dir
def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

# save data to pkl_dir
def save_pickle(pkl_dir, data):
    pickle.dump(data, open(pkl_dir, "wb"))
    

if __name__ == '__main__':
    root_dir = "../data_lx/datasets/4Dress"
    subj = "00185"
    outfit = "Inner"
    seq = "Take1"
    clothe = "Semantic/clothes"
    pkl_file = "cloth-f00011.pkl"

    garm_type = "lower"

    cano_smpl_path = os.path.join(root_dir, subj, "cano_smpl.ply")

    # save the extracted mesh data to a ply file
    extract_garm_mesh(root_dir, subj, outfit, seq)

    #create a ply file for the extracted mesh
    with open(os.path.join(root_dir, subj, outfit, seq, clothe, pkl_file), 'rb') as f:
        clothes_ply = pickle.load(f)
    garm_dict = clothes_ply[garm_type]
    skin_dict = clothes_ply['skin'] 

    # convert_dict_to_ply(mesh_dict, os.path.join(root_dir, subj, outfit, seq, "extracted_{}.ply".format(garm_type)))
    get_ref_smpl_offset(cano_smpl_path, garm_dict, skin_dict)