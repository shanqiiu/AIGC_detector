"""
Per-object SE(3) estimation on dynamic regions using PnP (global shutter only).
- Requires per-object masks (if unavailable, treat whole image as one object)
- Estimates object pose change between frames from 2D-2D flow + depth prior or plane homography fallback
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Dict, Optional, Tuple


class ObjectSE3Estimator:
    def __init__(self, ransac_reproj_thresh: float = 3.0, ransac_iter: int = 2000):
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.ransac_iter = ransac_iter

    def estimate_with_depth(self, flow: np.ndarray, depth: np.ndarray, K: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict:
        """Estimate SE(3) using PnP with depth lifting.
        Args:
            flow: HxWx2 optical flow from img1 to img2
            depth: HxW depth map at frame1
            K: 3x3 intrinsics
            mask: optional boolean HxW indicating object pixels
        Returns:
            dict with R,t, inliers, success flag
        """
        h, w = flow.shape[:2]
        if mask is None:
            mask = np.ones((h, w), dtype=bool)
        ys, xs = np.where(mask)
        if len(xs) < 30:
            return {'success': False}
        pts1 = np.stack([xs, ys], axis=1).astype(np.float32)
        pts2 = pts1 + flow[ys, xs]
        z = depth[ys, xs]
        valid = z > 1e-6
        if valid.sum() < 30:
            return {'success': False}
        pts1 = pts1[valid]
        pts2 = pts2[valid]
        z = z[valid]
        # back-project 3D
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        X = (pts1[:,0] - cx) * z / fx
        Y = (pts1[:,1] - cy) * z / fy
        Z = z
        obj_pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
        img_pts = pts2.reshape(-1, 1, 2).astype(np.float32)
        # PnP RANSAC
        dist = np.zeros((4,1), dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj_pts,
            imagePoints=img_pts,
            cameraMatrix=K,
            distCoeffs=dist,
            reprojectionError=self.ransac_reproj_thresh,
            iterationsCount=self.ransac_iter,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inliers is None or len(inliers) < 15:
            return {'success': False}
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3)
        return {
            'success': True,
            'R': R,
            't': t,
            'num_inliers': int(len(inliers)),
            'num_points': int(len(obj_pts)),
        }
