import os
import os.path as osp
import cv2
import numpy as np
import argparse
from demo_video.predictor import VisualizationDemo
import shutil

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for Demo')
    parser.add_argument('--video_frames_path', required=True, metavar='path to images', help='path to images')
    args = parser.parse_args()

    weights_path = './pretrained_models/trained_model.pth'
    cfg_file = './configs/hoist/hoistformer.yaml'
    demo_inference = VisualizationDemo(cfg_file, weights_path)

    root = args.video_frames_path
    video = root[root.rindex('/') + 1:]
    output_path = f'./masks/{video}'
    overlay_path = f'./overlays/{video}'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if os.path.exists(overlay_path):
        shutil.rmtree(overlay_path)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(overlay_path, exist_ok=True)
    videos = os.listdir(root)
    count = 0
    for vid_idx, vid in enumerate(videos):
        print(f'Processing video: {vid_idx}, total: {len(videos)}')
        frames_bgr_np = []
        frames = sorted(os.listdir(osp.join(root, vid)))
        for frm in frames:
            frm_np = cv2.imread(osp.join(root, vid, frm))
            frames_bgr_np.append(frm_np)
        frame_paths = [f'{output_path}/{frame}.png' for frame in frames]
        predictions, vis_output = demo_inference.run_on_video(frames_bgr_np, frame_paths)
        for frm_idx, vis_img in enumerate(vis_output):
            frm = vis_img.get_image()
            cv2.imwrite(f'{overlay_path}/frame{frm_idx + count}.jpg', frm)
        count += len(vis_output)
