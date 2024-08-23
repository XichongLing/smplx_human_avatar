import pickle
import numpy as np
from plyfile import PlyData, PlyElement
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import matplotlib.pyplot as plt
import trimesh
import pyrender
from PIL import Image
import open3d as o3d
import cv2
import tqdm
from pytorch3d.io import load_ply
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scipy import ndimage

from pytorch3d.transforms import axis_angle_to_matrix

from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    AmbientLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    SoftSilhouetteShader,
    SoftPhongShader,
    BlendParams,
    PerspectiveCameras,
)

import matplotlib.pyplot as plt
import cv2
import torch

def load_pickle(pkl_dir):
    return pickle.load(open(pkl_dir, "rb"))

def render_from_ply(root_dir, style, take, camera_idx):
    # parameters
    camera_type = "FoVPerspective"
    segmentation_type = "outer"

    basic_info = load_pickle(os.path.join(root_dir, style, take, 'basic_info.pkl'))
    scan_frames = basic_info['scan_frames']
    cameras = load_pickle(os.path.join(root_dir, style, take, "Capture/cameras.pkl"))
    camera_idx = '0004'
    if camera_type == "FoVPerspective":
        K = cameras[camera_idx]['intrinsics']
        extrinsics = cameras[camera_idx]['extrinsics']
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        # to be modified
        FovX = focal2fov(focal_length_x, 940)
        FovY = focal2fov(focal_length_y, 1280)
        R = extrinsics[:3, :3]
        T = extrinsics[:3, 3]
        R = R.transpose()
        R = np.expand_dims(R, axis=0)
        T = np.expand_dims(T, axis=0)
        camera = FoVPerspectiveCameras(
                device="cuda",
                R=R,
                T=T,
                fov=FovX,
                degrees=False,
                # aspect_ratio=0.734375,
        )
    elif camera_type == "perspective":
        img_path = os.path.join(root_dir, style, take, "Capture", camera_idx, "images/capture-f00099.png")
        # import ipdb; ipdb.set_trace()
        img = cv2.imread(img_path)
        image_shape = (img.shape[1], img.shape[0])
        intrinsic = torch.tensor((cameras[camera_idx]["intrinsics"]), dtype=torch.float32).cuda()
        extrinsic = torch.tensor(cameras[camera_idx]["extrinsics"], dtype=torch.float32).cuda()
        # assign camera image size
        image_size = torch.tensor([image_shape[0], image_shape[1]], dtype=torch.float32).unsqueeze(0).cuda()

        # assign camera parameters
        f_xy = torch.cat([intrinsic[0:1, 0], intrinsic[1:2, 1]], dim=0).unsqueeze(0)
        p_xy = intrinsic[:2, 2].unsqueeze(0)
        R = extrinsic[:, :3].unsqueeze(0)
        T = extrinsic[:, 3].unsqueeze(0)
        # coordinate system adaption to PyTorch3D
        R[:, :2, :] *= -1.0
        # camera position in world space -> world position in camera space
        T[:, :2] *= -1.0
        # R = torch.transpose(R, 1, 2)  # row-major
        camera = PerspectiveCameras(focal_length=f_xy, principal_point=p_xy, R=R, T=T, in_ndc=False, image_size=image_size).cuda()

    raster_settings = RasterizationSettings(
            image_size=(1280, 940),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
    )

    lights = AmbientLights(device="cuda")
    blend_params = BlendParams(background_color=(0.0, 0, 0))  
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device="cuda",
            cameras=camera,
            lights=lights,
        )
    )
    print('# # ============ Extract Labeled Clothes in Subj_Outfit_Seq: {}_{}_{} // Frames: {}'.format(root_dir, style, take, len(scan_frames)))
    # locate scan, label, cloth dir
    scan_dir = os.path.join(root_dir, style, take, 'Meshes_pkl')
    label_dir = os.path.join(root_dir, style, take, 'Semantic', 'labels')
    save_dir = os.path.join(root_dir, style, take, 'Rendered', 'Segmentation', segmentation_type, camera_idx)
    os.makedirs(save_dir, exist_ok=True)
    loop = tqdm.tqdm(range(len(scan_frames)))
    for n_frame in loop:
        # locate current frame
        frame = scan_frames[n_frame]
        label_path = os.path.join(label_dir, 'label-f{}.pkl'.format(frame))
        labels = np.array(load_pickle(label_path)['scan_labels']).astype(np.int32)
        if segmentation_type == "full":
            label_colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, 20))[:, :3]
            colors = torch.tensor(label_colors)[labels]
            vertex_colors = TexturesVertex(verts_features=colors[None].float()).to("cuda")
        elif segmentation_type == "outer":
            outer_color = torch.tensor([1., 0, 0])
            body_color = torch.tensor([0, 0, 1.])
            colors = torch.zeros(labels.shape[0], 3)
            colors[labels == 5] = outer_color 
            colors[labels != 5] = body_color  
            vertex_colors = TexturesVertex(verts_features=colors[None].float()).to("cuda")

        scan_path = os.path.join(scan_dir, 'mesh-f{}.pkl'.format(frame))    
        verts = torch.tensor(load_pickle(scan_path)['vertices'], ).cuda()
        faces = torch.tensor(load_pickle(scan_path)['faces']).cuda()
        mesh = Meshes(verts=[verts], faces=[faces])
        mesh.textures = vertex_colors


        images = renderer(mesh, cameras=camera, lights=lights, blend_params=blend_params)
        image_np = images[0, :, :, :3].cpu().numpy() * 255
        image_np = ndimage.rotate(image_np, 180)
        # image_np = cv2.resize(image_np, (940, 1280), interpolation=cv2.INTER_LINEAR)

        # gt_image = cv2.imread(gt_image_paths[index])
        # gt_image = cv2.resize(gt_image, (540, 540), interpolation=cv2.INTER_LANCZOS4)
        # gt_image_masked = np.where(image_np < 1, gt_image, 0)
        # cv2.imwrite("test.png", gt_image_masked)
        # cv2.imwrite("test_mask.png", image_np)
        save_path = os.path.join(save_dir, 'rendered_f{}.png'.format(frame))
        cv2.imwrite(save_path, image_np)

if __name__ == '__main__':
    # ply_path = "labeled_mesh-f00099.ply"
    # smpl_ply_path = "mesh-f00099_smpl.ply"
    # smpl_pkl_path = "mesh-f00099_smpl.pkl"
    # mesh_pkl = "Meshes_pkl/mesh-f00099.pkl"
    root_dir = "../data_lx/datasets/4Dress/00123/"
    style = "Outer"
    take = "Take11"
    camera_idx = '0004'
    # label_pkl = "Semantic/labels/label-f00099.pkl"
    # with open(os.path.join(root_dir,style,take,label_pkl), 'rb') as f_label, open(os.path.join(root_dir,style,take, mesh_pkl), 'rb') as f_ply:
    #     file_label = pickle.load(f_label)
    #     file_mesh = pickle.load(f_ply)  
    # render_from_ply_pyr(ply_path, root_dir, style, take)

    # render_from_ply(os.path.join(root_dir, style, take, 'SMPL', smpl_ply_path), root_dir, file_label, file_mesh, style, take, "rendered_image.png")
    render_from_ply(root_dir, style, take, camera_idx)