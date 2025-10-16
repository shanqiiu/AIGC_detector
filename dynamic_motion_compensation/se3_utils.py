"""
SE(3) utilities and camera projection helpers for global-shutter motion compensation.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3-vector.
    Args:
        v: shape (3,)
    Returns:
        3x3 skew-symmetric matrix
    """
    vx, vy, vz = v
    return np.array([[0.0, -vz, vy],
                     [vz, 0.0, -vx],
                     [-vy, vx, 0.0]], dtype=np.float64)


def se3_exp(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exponential map for SE(3) from twist (vx, vy, vz, wx, wy, wz).
    Args:
        xi: shape (6,), translation first then rotation
    Returns:
        R (3x3), t (3,)
    """
    v = xi[:3]
    w = xi[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        R = np.eye(3)
        t = v
        return R, t

    k = w / theta
    K = skew(k)
    s = np.sin(theta)
    c = np.cos(theta)
    R = np.eye(3) + s * K + (1.0 - c) * (K @ K)

    V = np.eye(3) + ((1.0 - c) / theta) * K + ((theta - s) / (theta**2)) * (K @ K)
    t = V @ v
    return R, t


def se3_compose(R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compose two SE(3): (R1,t1) o (R2,t2)."""
    R = R1 @ R2
    t = R1 @ t2 + t1
    return R, t


def project_points(K: np.ndarray, R: np.ndarray, t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project 3D points to pixels.
    Args:
        K: 3x3 intrinsics
        R, t: camera-to-world transform inverse (i.e., world to cam)
        P: Nx3 world points
    Returns:
        Nx2 pixel coordinates
    """
    Pc = (R @ P.T) + t.reshape(3, 1)
    x = (K @ Pc).T
    x = x[:, :2] / x[:, 2:3]
    return x


def grid_pixels(w: int, h: int) -> np.ndarray:
    """Return H*W x 2 grid of pixel coordinates (x,y)."""
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.stack([xs, ys], axis=-1).reshape(-1, 2)


def unproject_depth(K: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Back-project pixels with depth to 3D camera frame.
    Args:
        K: 3x3 intrinsics
        depth: HxW depth in meters (or consistent scale)
    Returns:
        N x 3 3D points in camera frame (z positive)
    """
    h, w = depth.shape
    uv = grid_pixels(w, h)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth.reshape(-1)
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    P = np.stack([x, y, z], axis=-1)
    return P


def homography_from_RTn(K: np.ndarray, R: np.ndarray, t: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Compute plane-induced homography: H = K (R - t n^T / d) K^{-1}.
    n is plane normal in camera-1 frame, d is plane distance (positive).
    """
    Kinv = np.linalg.inv(K)
    H = K @ (R - np.outer(t, n) / d) @ Kinv
    return H
