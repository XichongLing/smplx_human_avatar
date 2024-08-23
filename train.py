#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import fix_random, Evaluator, PSEvaluator
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss

import hydra
from omegaconf import OmegaConf
import wandb
import lpips


def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        # if value is a scalar, return itself
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value

def training(config):
    model = config.model
    dataset = config.dataset
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from
    save_video = config.save_video
    # define lpips
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda() # for training
    evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.train()

    gaussians.training_setup(opt)
    if checkpoint:
        scene.load_checkpoint(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    data_stack = None
    data_stack_len = len(scene.train_dataset)
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if save_video:
        height = 1280
        width = 940
        video_filename = 'rendered_video.mp4'
        segmentation_filename = 'segmentation_video.mp4'
        gt_filename = 'gt_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
        fps = 30  # Frames per second
        out_image = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
        out_segmentation = cv2.VideoWriter(segmentation_filename, fourcc, fps, (width, height))
        out_gt = cv2.VideoWriter(gt_filename, fourcc, fps, (width, height))
        frame_rate = 50

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        if iteration in [1,2, 500,501, 700, 701, 800, 801]:
            print("memory usage at line 100: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            # track the number of gaussians
            print(gaussians.get_xyz.shape)

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack)-1))
        data_t = data_idx / len(scene.train_dataset)
        data = scene.train_dataset[data_idx]

        if iteration in [1,2, 500,501, 700, 701, 800, 801]:
            print("memory usage at line 117: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        lambda_mask = C(iteration, config.opt.lambda_mask)
        use_mask = lambda_mask > 0.
        render_pkg = render(data, data_t, iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask, return_segmentation=True)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if iteration in [1,2, 500,501, 700, 701, 800, 801]: 
            print("memory usage at line 124: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)
        opacity = render_pkg["opacity_render"] if use_mask else None

        # if iteration in [300, 500, 700, 800]:
        #     import ipdb; ipdb.set_trace()
        # Loss
        gt_image = data.original_image.cuda()

        if save_video and iteration % frame_rate == 0: 
            # import ipdb; ipdb.set_trace()
            out_image_deepcopy = image.clone().detach()
            out_image_deepcopy = (out_image_deepcopy.permute(1,2,0).cpu().detach().numpy() * 255).astype(np.uint8)
            out_image.write(out_image_deepcopy)

            out_segmentation_deepcopy = render_pkg["segmentation_render"].clone().detach()
            out_segmentation_deepcopy = (out_segmentation_deepcopy.permute(1,2,0).cpu().detach().numpy() * 255).astype(np.uint8)
            # out_segmentation.write((render_pkg["segmentation_render"].permute(1,2,0).cpu().detach().numpy() * 255).astype(np.uint8))
            out_segmentation.write(out_segmentation_deepcopy)
            out_gt.write((data.original_image.permute(1,2,0).cpu().detach().numpy() * 255).astype(np.uint8))

        lambda_l1 = C(iteration, config.opt.lambda_l1)
        lambda_dssim = C(iteration, config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()

        if iteration in [1,2, 500,501, 700, 701, 800, 801]:
            print("memory usage at line 137: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image)
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image)  # ssim is a similarity metric, so we subtract it from 1 to get a loss

        # Here we can ignore the loss_dssim, since lambda_dssim is set to 0
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

        # lambda_l1_hands = C(iteration, config.opt.get('lambda_l1_hands', 1.))
        # lambda_dssim_hands = C(iteration, config.opt.get('lambda_dssim_hands', 0.))

        # left_hand_mask = render_pkg["left_hand_mask"]
        # left_hand_mask = torch.logical_and(left_hand_mask[0] != 0, torch.logical_and(left_hand_mask[1] != 0, left_hand_mask[2] != 0))
        # gt_left_hand_mask = data.left_hand_mask
        # gt_left_hand_mask = torch.Tensor(gt_left_hand_mask).to(left_hand_mask.device)
        # right_hand_mask = render_pkg["right_hand_mask"]
        # right_hand_mask = torch.logical_and(right_hand_mask[0] != 0, torch.logical_and(right_hand_mask[1] != 0, right_hand_mask[2] != 0))
        # gt_right_hand_mask = data.right_hand_mask
        # gt_right_hand_mask = torch.Tensor(gt_right_hand_mask).to(right_hand_mask.device)
        # loss_l1_hands = torch.tensor(0.).cuda()
        # loss_dssim_hands = torch.tensor(0.).cuda()

        # if (iteration > 4000):
        #     # Calculate L1 loss based on the hand masks
        #     if lambda_l1_hands > 0. or lambda_dssim_hands > 0.:
        #         # gt_left_hand_img = torch.where(gt_left_hand_mask != 0, gt_image, torch.zeros_like(gt_image))
        #         gt_left_hand_img = gt_left_hand_mask * gt_image
        #         gt_right_hand_img = gt_right_hand_mask * gt_image
        #         # gt_right_hand_img = torch.where(gt_right_hand_mask != 0, gt_image, torch.zeros_like(gt_image))
        #         # left_hand_img = torch.where(left_hand_mask != 0, image, torch.zeros_like(image))
        #         left_hand_img = left_hand_mask * image
        #         # right_hand_img = torch.where(right_hand_mask != 0, image, torch.zeros_like(image))
        #         right_hand_img = right_hand_mask * image
        #         if lambda_l1_hands > 0.:
        #             loss_l1_hands += (l1_loss(left_hand_img, gt_left_hand_img) + l1_loss(right_hand_img, gt_right_hand_img)) / 2
        #         if lambda_dssim_hands > 0.:
        #             loss_dssim_hands += (1.0 - ssim(left_hand_img, gt_left_hand_img) + 1.0 - ssim(right_hand_img, gt_right_hand_img)) / 2

        # loss += lambda_l1_hands * loss_l1_hands + lambda_dssim_hands * loss_dssim_hands

        # hand_mask = data.hand_mask.cuda()

        # maybe here use hand_mask
        # if (iteration > 2000): 
        #     #  randomly use mask
        #     hand_coeff = 1 + iteration / 10000
        #     loss += hand_coeff * lambda_l1 * l1_loss(image * hand_mask, gt_image * hand_mask)

        # perceptual loss

        if iteration in [1,2, 500,501, 700, 701, 800, 801]:
            print("memory usage at line 180: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)
        lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))
        if iteration in [1,2, 500,501, 700, 701, 800, 801]:
            print("memory usage at line 183: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)
        if lambda_perceptual > 0:
            # crop the foreground
            if iteration in [1,2, 500,501, 700, 701, 800, 801]:
                print("memory usage at line 187: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)
            mask = data.original_mask.cpu().numpy()
            mask = np.where(mask)
            y1, y2 = mask[1].min(), mask[1].max() + 1
            x1, x2 = mask[2].min(), mask[2].max() + 1

            # crop the image using the bounding box
            fg_image = image[:, y1:y2, x1:x2]
            gt_fg_image = gt_image[:, y1:y2, x1:x2]
            if iteration in [1,2, 500,501, 700, 701, 800, 801]:
                print("memory usage at line 196: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)
            # Perceptual loss, which is the difference between the two images as perceived by the human visual system,
            # is calculated using a pre-trained VGG network.
            loss_perceptual = loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
            if iteration in [1,2, 500,501, 700, 701, 800, 801]:
                print("memory usage at line 203: ", torch.cuda.memory_allocated(), " ,iteration: ", iteration)
            loss += lambda_perceptual * loss_perceptual
        else:
            loss_perceptual = torch.tensor(0.)

        # mask loss
        gt_mask = data.original_mask.cuda()
        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        elif config.opt.mask_loss_type == 'bce':
            opacity = torch.clamp(opacity, 1.e-3, 1.-1.e-3)
            loss_mask = F.binary_cross_entropy(opacity, gt_mask)
        elif config.opt.mask_loss_type == 'l1':
            loss_mask = F.l1_loss(opacity, gt_mask)
        else:
            raise ValueError
        loss += lambda_mask * loss_mask

        gt_segmentation = data.original_segmentation.cuda()
        loss_segmentation = F.l1_loss(render_pkg["segmentation_render"],gt_segmentation)
        loss += lambda_mask * loss_segmentation

        # mask_hands_loss
        # lambda_mask_hands = C(iteration, config.opt.get('lambda_mask_hands', 0.))
        # use_mask_hands = lambda_mask_hands > 0.

        # if not use_mask_hands:
        #     loss_mask_hands = torch.tensor(0.).cuda()
        # elif config.opt.mask_loss_type == 'bce':
        #     opacity = torch.clamp(opacity, 1.e-3, 1. - 1.e-3)
        #     left_hand_opacity = torch.where(left_hand_mask, opacity, torch.zeros_like(opacity)).squeeze(0)
        #     right_hand_opacity = torch.where(right_hand_mask, opacity, torch.zeros_like(opacity)).squeeze(0)
        #     loss_mask_hands = F.binary_cross_entropy(left_hand_opacity, gt_left_hand_mask) + F.binary_cross_entropy(right_hand_opacity, gt_right_hand_mask)
        #     loss_mask_hands /= 2
        # elif config.opt.mask_loss_type == 'l1':
        #     left_hand_opacity = torch.where(left_hand_mask, opacity, torch.zeros_like(opacity)).squeeze(0)
        #     right_hand_opacity = torch.where(right_hand_mask, opacity, torch.zeros_like(opacity)).squeeze(0)
        #     loss_mask_hands = F.l1_loss(left_hand_opacity, gt_left_hand_mask) + F.l1_loss(right_hand_opacity, gt_right_hand_mask)
        #     loss_mask_hands /= 2
        # else:
        #     raise ValueError
        # hand_mask_coeff = 1. + iteration / 10000   
        # loss_mask_hands *= hand_mask_coeff     
        # loss += lambda_mask_hands * loss_mask_hands

        # skinning loss

        # debuging memory usage
        # if iteration in [200, 500, 600, 700]:
        #     print("memory usage: ", torch.cuda.memory_allocated(), " iteration: ", iteration)
        lambda_skinning = C(iteration, config.opt.lambda_skinning)
        if lambda_skinning > 0:
            loss_skinning = scene.get_skinning_loss()  # Todo: Needed to inspect the skinning loss
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = C(iteration, config.opt.get('lambda_aiap_cov', 0.))
        if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
            # As-Isometric-As-Possible loss
            # Refer to the 3DGS paper
            # For each 3D Gaussian, we want to make sure the differences with neighbors in the deformed space
            # is as close as possible to the differences in the original space.
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
        loss += lambda_aiap_xyz * loss_aiap_xyz
        loss += lambda_aiap_cov * loss_aiap_cov

        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = opt.get(f"lambda_{name}", 0.)
            lbd = C(iteration, lbd)
            loss += lbd * value
        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                # 'loss/l1_hands_loss': loss_l1_hands.item(),
                # 'loss/ssim_hands_loss': loss_dssim_hands.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                'loss/segmentation_loss': loss_segmentation.item(),
                # 'loss/mask_hands_loss': loss_mask_hands.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            validation(iteration, testing_iterations, testing_interval, scene, evaluator,(pipe, background), data_t)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration > model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, scene, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                scene.optimize(iteration)

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)

    if save_video:
        out_image.release()
        out_segmentation.release()
        out_gt.release()

def validation(iteration, testing_iterations, testing_interval, scene : Scene, evaluator, renderArgs):
    # Report test and samples of training set
    if testing_interval > 0:
        # to record the first iteration
        if not iteration % testing_interval == 0 and iteration > 1:
            return
    else:
        if not iteration in testing_iterations:
            return

    scene.eval()
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : list(range(len(scene.test_dataset)))},
                          {'name': 'train', 'cameras' : [idx for idx in range(0, len(scene.train_dataset), len(scene.train_dataset) // 10)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            l1_hands_test = 0.0
            psnr_test = 0.0
            ssim_test = 0.0
            lpips_test = 0.0
            psnr_hands_test = 0.0
            ssim_hands_test = 0.0
            lpips_hands_test = 0.0
            examples = []
            for idx, data_idx in enumerate(config['cameras']):
                data = getattr(scene, config['name'] + '_dataset')[data_idx]
                data_t = data_idx / len(getattr(scene, config['name'] + '_dataset'))
                render_pkg = render(data, data_t, iteration, scene, *renderArgs, compute_loss=False, return_opacity=True, return_segmentation=True)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(data.original_image.to("cuda"), 0.0, 1.0)
                opacity_image = torch.clamp(render_pkg["opacity_render"], 0.0, 1.0)
                segmentation_image = render_pkg["segmentation_render"]
                # import ipdb; ipdb.set_trace()


                wandb_img = wandb.Image(opacity_image[None],
                                        caption=config['name'] + "_view_{}/render_opacity".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(image[None], caption=config['name'] + "_view_{}/render".format(data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(gt_image[None], caption=config['name'] + "_view_{}/ground_truth".format(
                    data.image_name))
                examples.append(wandb_img)
                wandb_img = wandb.Image(segmentation_image[None], caption=config['name'] + "_view_{}/segmentation".format(data.image_name))
                examples.append(wandb_img)

                l1_test += l1_loss(image, gt_image).mean().double()
                metrics_test = evaluator(image, gt_image)
                psnr_test += metrics_test["psnr"]
                ssim_test += metrics_test["ssim"]
                lpips_test += metrics_test["lpips"]

                # hands_mask = torch.clamp(render_pkg["left_hand_mask"] + render_pkg["right_hand_mask"], 0.0, 1.0)
                # image_hands = torch.where(hands_mask != 0, image, torch.zeros_like(image))
                # gt_hands_mask = torch.Tensor(data.left_hand_mask + data.right_hand_mask).to(hands_mask.device)
                # gt_image_hands = torch.where(gt_hands_mask != 0, gt_image, torch.zeros_like(gt_image))
                # l1_hands_test += l1_loss(image_hands, gt_image_hands).mean().double()
                # metrics_test = evaluator(image_hands, gt_image_hands)
                # psnr_hands_test += metrics_test["psnr"]
                # ssim_hands_test += metrics_test["ssim"]
                # lpips_hands_test += metrics_test["lpips"]

                wandb.log({config['name'] + "_images": examples})
                examples.clear()

            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            # psnr_hands_test /= len(config['cameras'])
            # ssim_hands_test /= len(config['cameras'])
            # lpips_hands_test /= len(config['cameras'])
            # l1_hands_test /= len(config['cameras'])
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            wandb.log({
                config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                config['name'] + '/loss_viewpoint - psnr': psnr_test,
                config['name'] + '/loss_viewpoint - ssim': ssim_test,
                config['name'] + '/loss_viewpoint - lpips': lpips_test,
                # config['name'] + '/loss_hands - l1_loss': l1_hands_test,
                # config['name'] + '/loss_hands - psnr': psnr_hands_test,
                # config['name'] + '/loss_hands - ssim': ssim_hands_test,
                # config['name'] + '/loss_hands - lpips': lpips_hands_test,
            })

    wandb.log({'scene/opacity_histogram': wandb.Histogram(scene.gaussians.get_opacity.cpu())})
    wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]})
    torch.cuda.empty_cache()
    scene.train()

def video_writing(images, video_path):
    # Assuming final_tensor is your tensor of shape [100, 1280, 940, 3]

    # Normalize the tensor if it's not already in [0, 1]
    images = images - images.min()
    images = images / images.max()

    # Convert the tensor to uint8 and move to CPU if necessary
    images = (images * 255).to(torch.uint8)

    # Convert the tensor to a NumPy array
    PIL_list = images.cpu().numpy()

    # Define the video codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID', 'DIVX', etc.
    fps = 10  # Frames per second
    height, width, _ = PIL_list[0].shape
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for i in range(PIL_list.shape[0]):
        frame = PIL_list[i]
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print(f'Video saved as {video_path}')

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False) # allow adding new values to config

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)

    # set wandb logger
    wandb_name = config.name
    wandb.init(
        # mode="disabled",
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        # project='gaussian-splatting-avatar',
        project='3dgs',
        entity='gsgarm',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )



    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
