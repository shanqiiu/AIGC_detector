# -*- coding: utf-8 -*-
"""
CLI for global-shutter multi-view motion compensation focused on dynamic objects.
Steps:
- Read two consecutive frames or a folder
- Compute optical flow using existing RAFT predictor in repo
- Estimate camera homography and subtract camera-induced flow
- (Optional) Estimate per-object SE3 if depth and masks are provided
- Save residual flows and a simple report under the new folder
"""
from __future__ import annotations
import os
import glob
import json
import argparse
from typing import Optional, List, Dict
import numpy as np
import cv2

try:
    from raft_model import RAFTPredictor
except Exception:
    from simple_raft import SimpleRAFTPredictor as RAFTPredictor

from dynamic_motion_compensation.camera_compensation import CameraCompensator
from dynamic_motion_compensation.object_motion import ObjectSE3Estimator


def load_images_from_dir(img_dir: str, max_frames: Optional[int] = None, frame_skip: int = 1) -> List[np.ndarray]:
    files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
        files.extend(glob.glob(os.path.join(img_dir, ext)))
    files.sort()
    frames = []
    for i, f in enumerate(files):
        if i % frame_skip != 0:
            continue
        if max_frames is not None and len(frames) >= max_frames:
            break
        img = cv2.imread(f)
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def estimate_intrinsics_from_fov(shape, fov_deg: float) -> np.ndarray:
    h, w = shape[:2]
    f = w / (2 * np.tan(np.deg2rad(fov_deg / 2)))
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
    return K


def main():
    parser = argparse.ArgumentParser(description='Global-shutter dynamic motion compensation CLI')
    parser.add_argument('--input', '-i', required=True, help='Video file or image folder')
    parser.add_argument('--output', '-o', required=True, help='Output folder')
    parser.add_argument('--device', default='cpu', help='cuda or cpu')
    parser.add_argument('--raft_model', default=None, help='Optional RAFT weights path')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--fov', type=float, default=60.0, help='Horizontal FOV degrees for intrinsics')
    parser.add_argument('--depth', type=str, default=None, help='Optional depth npy file or folder aligned to frames')
    parser.add_argument('--masks', type=str, default=None, help='Optional masks npy file/folder for per-object SE3')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load frames
    frames: List[np.ndarray]
    if os.path.isdir(args.input):
        frames = load_images_from_dir(args.input, args.max_frames, args.frame_skip)
    else:
        cap = cv2.VideoCapture(args.input)
        frames = []
        idx = 0
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if idx % args.frame_skip == 0:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                if args.max_frames is not None and len(frames) >= args.max_frames:
                    break
            idx += 1
        cap.release()

    if len(frames) < 2:
        raise RuntimeError('Need at least two frames')

    # Intrinsics
    K = estimate_intrinsics_from_fov(frames[0].shape, args.fov)

    # Predict flow sequence
    raft = RAFTPredictor(args.raft_model, args.device)
    flows = []
    for i in range(len(frames) - 1):
        f = raft.predict_flow(frames[i], frames[i+1])
        if f.ndim == 3 and f.shape[0] == 2:
            f = np.transpose(f, (1, 2, 0))
        flows.append(f.astype(np.float32))

    # Compensate camera motion
    comp = CameraCompensator()
    residuals = []
    cam_flows = []
    homographies = []
    stats = []
    for i in range(len(flows)):
        out = comp.compensate(flows[i], frames[i], frames[i+1])
        residuals.append(out['residual_flow'])
        cam_flows.append(out['camera_flow'])
        homographies.append(out['homography'])
        mag = np.linalg.norm(out['residual_flow'].reshape(-1, 2), axis=1)
        stats.append({'frame': i, 'residual_mean': float(mag.mean()), 'residual_q90': float(np.quantile(mag, 0.90))})

    # Optional: per-object SE3 if depth+masks provided (not required)
    obj_results = []
    if args.depth is not None and args.masks is not None:
        est = ObjectSE3Estimator()
        # Load depth and masks per frame index
        def load_frame_asset(path, idx):
            # supports npy per frame or single npy with stack
            if os.path.isdir(path):
                fname = os.path.join(path, f'{idx:04d}.npy')
                return np.load(fname)
            arr = np.load(path)
            return arr[idx]
        for i in range(len(flows)):
            depth1 = load_frame_asset(args.depth, i)
            mask = load_frame_asset(args.masks, i).astype(bool)
            res = est.estimate_with_depth(flows[i], depth1, K, mask)
            obj_results.append(res)

    # Save outputs
    np.save(os.path.join(args.output, 'flows.npy'), np.array(flows, dtype=object))
    np.save(os.path.join(args.output, 'camera_flows.npy'), np.array(cam_flows, dtype=object))
    np.save(os.path.join(args.output, 'residual_flows.npy'), np.array(residuals, dtype=object))
    with open(os.path.join(args.output, 'stats.json'), 'w') as f:
        json.dump({'per_frame': stats}, f, indent=2)

    # Simple visualizations for first few frames
    vis_dir = os.path.join(args.output, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    def vis_mag(m):
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        return (cv2.applyColorMap((m*255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1])
    for i in range(min(5, len(flows))):
        mag0 = np.linalg.norm(flows[i], axis=2)
        mag1 = np.linalg.norm(residuals[i], axis=2)
        cv2.imwrite(os.path.join(vis_dir, f'flow_mag_{i:04d}.png'), vis_mag(mag0))
        cv2.imwrite(os.path.join(vis_dir, f'residual_mag_{i:04d}.png'), vis_mag(mag1))

    print(f'Saved outputs to {args.output}')


if __name__ == '__main__':
    main()
