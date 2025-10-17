"""
Global-shutter camera motion compensation and residual flow computation.
- Estimate planar/global homography from feature matches
- Compute camera-induced flow and subtract from RAFT flow
- Optionally use plane-induced homography with known plane normal
"""
from __future__ import annotations
import numpy as np
import cv2
from typing import Optional, Dict, Tuple


class CameraCompensator:
    def __init__(self, ransac_thresh: float = 1.0, ransac_iters: int = 2000, feature: str = 'ORB', max_features: int = 2000):
        if feature == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif feature == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            self.detector = cv2.ORB_create(nfeatures=max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ransac_thresh = ransac_thresh
        self.ransac_iters = ransac_iters

    def estimate_homography(self, img1: np.ndarray, img2: np.ndarray, min_matches: int = 12) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2
        k1, d1 = self.detector.detectAndCompute(g1, None)
        k2, d2 = self.detector.detectAndCompute(g2, None)
        if d1 is None or d2 is None:
            return None, None
        matches = self.matcher.match(d1, d2)
        if len(matches) < min_matches:
            return None, None
        matches = sorted(matches, key=lambda m: m.distance)[:max(min_matches*5, 200)]
        pts1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, self.ransac_thresh, maxIters=self.ransac_iters)
        return H, mask

    def camera_flow_from_homography(self, H: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        y, x = np.mgrid[0:h, 0:w]
        coords = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)  # (N,3)
        tc = (H @ coords.T).T
        tc = tc[:, :2] / tc[:, 2:3]
        cam_flow = tc - coords[:, :2]
        return cam_flow.reshape(h, w, 2)

    def compensate(self, flow: np.ndarray, img1: np.ndarray, img2: np.ndarray) -> Dict:
        H, mask = self.estimate_homography(img1, img2)
        if H is None:
            return {
                'homography': None,
                'camera_flow': np.zeros_like(flow),
                'residual_flow': flow,
                'inliers': 0,
                'total_matches': 0,
            }
        cam_flow = self.camera_flow_from_homography(H, flow.shape[:2])
        residual = flow - cam_flow
        inliers = int(mask.sum()) if mask is not None else 0
        return {
            'homography': H,
            'camera_flow': cam_flow,
            'residual_flow': residual,
            'inliers': inliers,
            'total_matches': int(mask.size) if mask is not None else 0,
        }
