import pickle
import numpy as np
from plyfile import PlyData, PlyElement
import os

def face_coloring(faces, labels):
    faces_colored = [tuple(face, labels[face[0]] * 255, 255, 255) for face in faces]
def create_ply_plyfile(pickle_file, file_path):
    # Open a new PLY file for writing
    vertices = np.array(pickle_file['vertices'], )
    vertices = [tuple(vertex) for vertex in vertices]
    vertices_plyfile = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    faces = np.array(pickle_file['faces'])
    faces = faces.astype(np.int32)
    faces = [tuple([face]) for face in faces]
    print(faces)
    faces_plyfile = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
    plydata = PlyData(
        [
            PlyElement.describe(vertices_plyfile, 'vertex'),
            PlyElement.describe(faces_plyfile, 'face')
        ],
        text=True
    )
    plydata.write(file_path)

def create_labelled_ply(file_mesh, file_label, save_path):
    labels = np.array(file_label['scan_labels'])

    vertices = np.array(file_mesh['vertices'], )
    vertices = [tuple(vertex) for vertex in vertices]
    vertices_plyfile = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    faces = np.array(file_mesh['faces'])
    faces = faces.astype(np.int32)
    # faces = [tuple([face]) for face in faces]
    faces_colored = [(list(face), labels[face[0]] * 255, 255, 255) for face in faces]
    faces_plyfile = np.array(faces_colored, dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])


    plydata = PlyData(
        [
            PlyElement.describe(vertices_plyfile, 'vertex'),
            PlyElement.describe(faces_plyfile, 'face'),
        ],
        text=True
    )
    plydata.write(save_path)


if __name__ == '__main__':
    root_dir = "../../project_lx/datasets/4Dress/00123/"
    style = "Outer"
    take = "Take9"
    mesh_pkl = "Meshes_pkl/mesh-f00099.pkl"
    label_pkl = "Semantic/labels/label-f00099.pkl"
    save_path = "labeled_mesh-f00099.ply"
    with open(os.path.join(root_dir,style,take,mesh_pkl), 'rb') as f_mesh, open(os.path.join(root_dir,style,take,label_pkl), 'rb') as f_label:
        file_mesh = pickle.load(f_mesh)
        file_label = pickle.load(f_label)
    create_labelled_ply(file_mesh, file_label, save_path)

