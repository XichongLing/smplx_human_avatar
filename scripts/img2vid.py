import os
import cv2
import numpy as np
import torch
import tqdm

DATASET_DIR = "../data_lx/datasets/4Dress"
IMG_HEIGHT = 1280
IMG_WIDTH = 940


def video_generation(data_dir, video_path, img_size=(IMG_WIDTH, IMG_HEIGHT), ):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    fps = 30  # Frames per second
    out_video = cv2.VideoWriter(video_path, fourcc, fps, img_size)
    sorted_files = sorted(os.listdir(data_dir))
    loop = tqdm.tqdm(range(len(sorted_files)))
    for img_idx in loop:
        img_name = sorted_files[img_idx]
        loop.set_description("writing images to video: {}/{}".format(img_idx, len(sorted_files)))
        img = cv2.imread(os.path.join(data_dir, img_name))
        out_video.write(img.copy())
    out_video.release()


if __name__ == "__main__":
    subj_dict = ['00123', '00185', '00187']
    outfit_dict = ['Outer', 'Inner']
    seq_dict = ['Take1', 'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take8', 'Take9', 'Take10', 'Take11', 'Take12', 'Take13', 'Take14', 'Take15', 'Take16', 'Take17', 'Take18', 'Take19', 'Take20']
    camera_dict = ['0004', '0028', '0052', '0076']
    for seq_idx in range(7,13):
        for cam in camera_dict:
            data_dir = os.path.join(DATASET_DIR, subj_dict[0], outfit_dict[0], seq_dict[seq_idx],"Capture", cam, "images")
            video_filename = "concatenated-video.mp4"
            video_dir = os.path.join(DATASET_DIR, subj_dict[0], outfit_dict[0], seq_dict[seq_idx], "Capture", cam, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, video_filename)
            video_generation(data_dir, video_path, (940, 1280))