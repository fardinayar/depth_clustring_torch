"""High-level clustering API for the three common inputs.

    cluster_disparity(...)     stereo disparity image  -> labels in pixel space
    cluster_range_image(...)   organized range image   -> labels in range-image space
    cluster_point_cloud(...)   arbitrary 3D points     -> per-point labels

All return relabelled int labels (consecutive ids, -1 = background/invalid).
"""
import math
import torch

from .clustering import (
    disparity_to_range, depth_to_range, pinhole_alphas, build_alphas,
    remove_ground, points_to_spherical, cluster,
)


def cluster_disparity(disparity, fx, baseline, cx, cy, fy=None, *,
                      theta_deg=7.0, min_size=0, ground=False,
                      ground_thresh_deg=20.0):
    """Stereo disparity -> Euclidean range (perspective) -> cluster.

    disparity: (H,W) or (B,H,W) in pixels. Labels come back pixel-aligned with
    the disparity/RGB image (no back-projection needed)."""
    fy = fx if fy is None else fy
    range_img, valid = disparity_to_range(disparity, fx, baseline, cx, cy, fy)
    B, H, W = range_img.shape
    if ground:
        v = torch.arange(H, dtype=torch.float32)
        row_elev = torch.atan((v - cy) / fy)
        g = remove_ground(range_img, valid, row_elev,
                          ground_angle_thresh=math.radians(ground_thresh_deg))
        valid = valid & ~g
    row_a, col_a = pinhole_alphas(H, W, fx, fy, cx, cy)
    return cluster(range_img, valid, row_a, col_a,
                   threshold=math.radians(theta_deg), wrap=False,
                   relabel=True, min_size=min_size)


def cluster_range_image(range_img, valid=None, *, row_angles=None, col_angles=None,
                        row_alphas=None, col_alphas=None, theta_deg=7.0,
                        wrap=True, min_size=0, ground=False, ground_thresh_deg=5.0):
    """Cluster an already-organized range image (e.g. native LiDAR).

    Provide either the sensor angle tables (row_angles (H,), col_angles (W,)) or
    precomputed (row_alphas, col_alphas). Labels are in range-image space."""
    if range_img.dim() == 2:
        range_img = range_img.unsqueeze(0)
    B, H, W = range_img.shape
    if valid is None:
        valid = range_img > 0
    if valid.dim() == 2:
        valid = valid.unsqueeze(0)
    if ground and row_angles is not None:
        g = remove_ground(range_img, valid, row_angles,
                          ground_angle_thresh=math.radians(ground_thresh_deg))
        valid = valid & ~g
    if row_alphas is None or col_alphas is None:
        if row_angles is None or col_angles is None:
            raise ValueError("Provide row_angles/col_angles or row_alphas/col_alphas")
        row_alphas, col_alphas = build_alphas(row_angles, col_angles, wrap=wrap)
    return cluster(range_img, valid, row_alphas, col_alphas,
                   threshold=math.radians(theta_deg), wrap=wrap,
                   relabel=True, min_size=min_size)


def cluster_point_cloud(points, *, n_rows=64, n_cols=1024, theta_deg=7.0,
                        wrap=True, min_size=0, fov_up_deg=None, fov_down_deg=None,
                        valid=None, return_range_labels=False, ground=True, ground_thresh_deg=5.0):
    """Project a 3D point cloud (x-forward, y-left, z-up) into a spherical range
    image, cluster, and scatter labels back to the points.

    points: (N,3) or (B,N,3). Returns per-point labels (B,N); if
    return_range_labels, also the (B,Hs,Ws) range-image labels."""
    sp = points_to_spherical(points, n_rows, n_cols, valid=valid, wrap=wrap,
                             fov_up_deg=fov_up_deg, fov_down_deg=fov_down_deg)
    if ground:
        if fov_up_deg is None or fov_down_deg is None:
            raise ValueError("fov_up_deg and fov_down_deg must be explicitly provided for ground removal.")
        
        row_elev = torch.linspace(math.radians(fov_up_deg), 
                                  math.radians(fov_down_deg), 
                                  n_rows, 
                                  device=points.device, 
                                  dtype=torch.float32)
        
        g = remove_ground(sp["sph_range"], sp["valid_sph"], row_elev,
                          ground_angle_thresh=math.radians(ground_thresh_deg))
        sp["valid_sph"] = sp["valid_sph"] & ~g
    labels_sph = cluster(sp["sph_range"], sp["valid_sph"],
                         sp["row_alphas"], sp["col_alphas"],
                         threshold=math.radians(theta_deg), wrap=wrap,
                         relabel=True, min_size=min_size)
    B = labels_sph.shape[0]
    binidx = sp["bin"]                                       # (B,N)
    flat = labels_sph.view(B, -1)
    point_labels = torch.full_like(binidx, -1, dtype=labels_sph.dtype)
    for b in range(B):
        point_labels[b] = flat[b][binidx[b]]
    point_labels = torch.where(sp["valid"], point_labels,
                               torch.full_like(point_labels, -1))
    if return_range_labels:
        return point_labels, labels_sph
    return point_labels
