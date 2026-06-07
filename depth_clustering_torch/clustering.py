"""Fast batched range-image angle clustering for PyTorch.

Public API
----------
build_alphas(row_angles, col_angles, wrap)  -> (row_alphas, col_alphas)
project(points, row_angles, W, ...)         -> (range_img, valid, px_r, px_c)
remove_ground(range_img, row_angles, ...)   -> ground mask
cluster(range_img, valid, row_alphas, col_alphas, threshold, wrap, relabel)
cluster_reference(...)                       -> pure-torch label propagation (slow, for tests)
"""

import os
import math
import torch

# --------------------------------------------------------------------------
# CUDA extension (JIT-compiled on first import)
# --------------------------------------------------------------------------

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    # Make sure nvcc is discoverable for torch's JIT compiler.
    cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    os.environ.setdefault("CUDA_HOME", cuda_home)
    nvcc_bin = os.path.join(cuda_home, "bin")
    if nvcc_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = nvcc_bin + os.pathsep + os.environ.get("PATH", "")

    from torch.utils.cpp_extension import load
    src = os.path.join(os.path.dirname(__file__), "csrc", "cc_cuda.cu")
    _EXT = load(
        name="depth_cluster_cuda",
        sources=[src],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _EXT


# --------------------------------------------------------------------------
# alpha tables
# --------------------------------------------------------------------------

def build_alphas(row_angles, col_angles, wrap=True):
    """Build per-edge angular steps consumed by the kernel.

    row_angles: (H,) vertical beam angles (rad), increasing or decreasing.
    col_angles: (W,) azimuth angles (rad).
    Returns (row_alphas (H,), col_alphas (W,)). The last entry of each is the
    step to the *next* row/col; col_alphas[W-1] holds the wrap step (col W-1->0).
    """
    row_angles = torch.as_tensor(row_angles, dtype=torch.float32)
    col_angles = torch.as_tensor(col_angles, dtype=torch.float32)
    H, W = row_angles.numel(), col_angles.numel()

    row_alphas = torch.empty(H, dtype=torch.float32)
    row_alphas[:-1] = (row_angles[1:] - row_angles[:-1]).abs()
    row_alphas[-1] = row_alphas[-2] if H > 1 else 0.0

    col_alphas = torch.empty(W, dtype=torch.float32)
    col_alphas[:-1] = (col_angles[1:] - col_angles[:-1]).abs()
    if wrap:
        span = (col_angles[-1] - col_angles[0]).abs()
        full = 2.0 * math.pi
        col_alphas[-1] = (full - span).abs() / max(W - 1, 1) if W > 1 else 0.0
    else:
        col_alphas[-1] = col_alphas[-2] if W > 1 else 0.0
    return row_alphas, col_alphas


# --------------------------------------------------------------------------
# disparity / depth -> Euclidean range image (pure-torch, vectorized)
# --------------------------------------------------------------------------

def _range_factor(H, W, fx, fy, cx, cy, device, dtype=torch.float32):
    """Precompute sqrt(xn^2 + yn^2 + 1) over the pixel grid (a (H,W) map)."""
    v = torch.arange(H, device=device, dtype=dtype)
    u = torch.arange(W, device=device, dtype=dtype)
    yn = ((v - cy) / fy).view(H, 1)
    xn = ((u - cx) / fx).view(1, W)
    return torch.sqrt(xn * xn + yn * yn + 1.0)


def disparity_to_range(disparity, fx, baseline, cx, cy, fy=None, min_disp=1e-3):
    """Convert a stereo disparity image to a Euclidean range image.

    disparity: (H,W) or (B,H,W) in pixels. fx/fy in pixels, baseline in metres.
    Returns (range_img, valid) with the same leading shape; range in metres,
    valid = disparity > min_disp. Fully vectorized; runs wherever the tensor lives.
    """
    if disparity.dim() == 2:
        disparity = disparity.unsqueeze(0)
    B, H, W = disparity.shape
    fy = fx if fy is None else fy
    factor = _range_factor(H, W, fx, fy, cx, cy, disparity.device, disparity.dtype)
    valid = disparity > min_disp
    z = (fx * baseline) / disparity.clamp_min(min_disp)      # depth along axis
    range_img = z * factor                                   # Euclidean range
    range_img = torch.where(valid, range_img, torch.zeros_like(range_img))
    return range_img, valid


def depth_to_range(depth, fx, cx, cy, fy=None):
    """Convert a perspective depth image (Z along optical axis) to Euclidean range."""
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    B, H, W = depth.shape
    fy = fx if fy is None else fy
    factor = _range_factor(H, W, fx, fy, cx, cy, depth.device, depth.dtype)
    valid = depth > 0
    return torch.where(valid, depth * factor, torch.zeros_like(depth)), valid


def pinhole_alphas(H, W, fx, fy, cx, cy):
    """Per-axis angular steps for a pinhole camera (separable approximation),
    ready to pass as (row_alphas, col_alphas) to cluster() with wrap=False."""
    v = torch.arange(H + 1, dtype=torch.float32)
    u = torch.arange(W + 1, dtype=torch.float32)
    phi = torch.atan((v - cy) / fy)          # vertical ray angle per row edge
    theta = torch.atan((u - cx) / fx)        # horizontal ray angle per col edge
    row_alphas = (phi[1:] - phi[:-1]).abs()  # (H,)
    col_alphas = (theta[1:] - theta[:-1]).abs()  # (W,)
    return row_alphas, col_alphas


# --------------------------------------------------------------------------
# perspective depth -> spherical (LiDAR-style) range image + pixel->bin mapping
# --------------------------------------------------------------------------

def depth_to_spherical(depth, valid, fx, fy, cx, cy,
                       n_rows=None, n_cols=None, eps=1e-9):
    """Re-bin a perspective depth image into a spherical (elevation x azimuth)
    range image, exactly as the original depth_clustering organizes LiDAR.

    Returns a dict with:
      sph_range (B,Hs,Ws) Euclidean range per spherical bin (0 where empty)
      valid_sph (B,Hs,Ws) bool occupancy
      row, col  (B,H,W) long  spherical bin each *camera pixel* falls into
                              (use to back-project labels: lab_px = lab_sph[row,col])
      count     (B,Hs,Ws) float  #camera pixels per bin (collisions if >1)
      row_alphas (Hs,), col_alphas (Ws,)  uniform angular steps for cluster()
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
        valid = valid.unsqueeze(0)
    B, H, W = depth.shape
    dev = depth.device
    n_rows = n_rows or H
    n_cols = n_cols or W

    v = torch.arange(H, device=dev, dtype=torch.float32).view(1, H, 1)
    u = torch.arange(W, device=dev, dtype=torch.float32).view(1, 1, W)
    Z = depth
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    r = torch.sqrt(X * X + Y * Y + Z * Z)
    az = torch.atan2(X, Z)                                   # depends on column
    el = torch.atan2(Y, torch.sqrt(X * X + Z * Z).clamp_min(eps))  # column+row

    big = torch.tensor(1e9, device=dev)
    az_min = torch.where(valid, az, big).amin()
    az_max = torch.where(valid, az, -big).amax()
    el_min = torch.where(valid, el, big).amin()
    el_max = torch.where(valid, el, -big).amax()

    col = ((az - az_min) / (az_max - az_min + eps) * n_cols).floor().long().clamp(0, n_cols - 1)
    row = ((el - el_min) / (el_max - el_min + eps) * n_rows).floor().long().clamp(0, n_rows - 1)
    flat = row * n_cols + col                                # (B,H,W)

    sph = torch.full((B, n_rows, n_cols), float("inf"), device=dev)
    count = torch.zeros((B, n_rows, n_cols), device=dev)
    for b in range(B):
        m = valid[b]
        sph[b].view(-1).scatter_reduce_(0, flat[b][m], r[b][m], reduce="amin", include_self=True)
        count[b].view(-1).scatter_add_(0, flat[b][m], torch.ones_like(r[b][m]))
    valid_sph = torch.isfinite(sph)
    sph = torch.where(valid_sph, sph, torch.zeros_like(sph))

    row_alphas = torch.full((n_rows,), float((el_max - el_min) / n_rows))
    col_alphas = torch.full((n_cols,), float((az_max - az_min) / n_cols))
    return dict(sph_range=sph, valid_sph=valid_sph, row=row, col=col,
                count=count, row_alphas=row_alphas, col_alphas=col_alphas)


def points_to_spherical(points, n_rows, n_cols, valid=None, wrap=True,
                        fov_up_deg=None, fov_down_deg=None, eps=1e-9):
    """Project an arbitrary 3D point cloud into a spherical (elevation x azimuth)
    range image, the native organization of LiDAR for depth_clustering.

    points: (N,3) or (B,N,3), convention x-forward, y-left, z-up.
    Returns a dict with sph_range (B,Hs,Ws), valid_sph, per-point `bin` (B,N) and
    `valid` (B,N) for scattering labels back to points, and row/col_alphas."""
    if points.dim() == 2:
        points = points.unsqueeze(0)
    B, N, _ = points.shape
    dev = points.device
    if valid is None:
        valid = torch.ones(B, N, dtype=torch.bool, device=dev)
    elif valid.dim() == 1:
        valid = valid.unsqueeze(0)

    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    r = torch.sqrt(x * x + y * y + z * z)
    valid = valid & (r > eps)
    az = torch.atan2(y, x)                                   # azimuth, [-pi, pi]
    el = torch.atan2(z, torch.sqrt(x * x + y * y).clamp_min(eps))  # elevation

    if fov_up_deg is not None and fov_down_deg is not None:
        el_max, el_min = math.radians(fov_up_deg), math.radians(fov_down_deg)
    else:
        el_max = float(torch.where(valid, el, torch.full_like(el, -1e9)).amax())
        el_min = float(torch.where(valid, el, torch.full_like(el, 1e9)).amin())
    if wrap:
        az_min, az_max = -math.pi, math.pi
    else:
        az_max = float(torch.where(valid, az, torch.full_like(az, -1e9)).amax())
        az_min = float(torch.where(valid, az, torch.full_like(az, 1e9)).amin())
    el_span = max(el_max - el_min, eps)
    az_span = max(az_max - az_min, eps)

    row = (((el_max - el) / el_span) * n_rows).floor().long().clamp(0, n_rows - 1)
    col = (((az - az_min) / az_span) * n_cols).floor().long().clamp(0, n_cols - 1)
    binidx = row * n_cols + col                             # (B,N)

    sph = torch.full((B, n_rows, n_cols), float("inf"), device=dev)
    count = torch.zeros((B, n_rows, n_cols), device=dev)
    for b in range(B):
        m = valid[b]
        sph[b].view(-1).scatter_reduce_(0, binidx[b][m], r[b][m], reduce="amin", include_self=True)
        count[b].view(-1).scatter_add_(0, binidx[b][m], torch.ones_like(r[b][m]))
    valid_sph = torch.isfinite(sph)
    sph = torch.where(valid_sph, sph, torch.zeros_like(sph))

    row_alphas = torch.full((n_rows,), float(el_span / n_rows))
    col_alphas = torch.full((n_cols,), float(az_span / n_cols))
    return dict(sph_range=sph, valid_sph=valid_sph, bin=binidx, valid=valid,
                count=count, row_alphas=row_alphas, col_alphas=col_alphas)


def backproject_labels(labels_sph, row, col, valid):
    """Scatter spherical-bin labels back to camera pixel space.
    labels_sph (B,Hs,Ws) int; row/col/valid (B,H,W). Returns labels_px (B,H,W)."""
    B, Hs, Ws = labels_sph.shape
    flat = (row * Ws + col)                                  # (B,H,W)
    out = torch.full_like(row, -1, dtype=labels_sph.dtype)
    for b in range(B):
        out[b] = labels_sph[b].view(-1)[flat[b]]
    out = torch.where(valid, out, torch.full_like(out, -1))
    return out


# --------------------------------------------------------------------------
# projection: point cloud -> range image
# --------------------------------------------------------------------------

def project(points, row_angles, W, fov_start=-math.pi, fov_end=math.pi, batch_index=None):
    """Project xyz points into a (B,H,W) range image.

    points: (N,3) or (B,N,3). row_angles: (H,) vertical beam angles (rad).
    Nearest beam row is chosen per point; nearest point (min range) wins a pixel.
    Returns range_img (B,H,W), valid (B,H,W) bool, and the (row,col) of each point.
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)
    B, N, _ = points.shape
    H = row_angles.numel()
    row_angles = row_angles.to(points.device, torch.float32)
    dev = points.device

    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    rng = torch.sqrt(x * x + y * y + z * z)
    horiz = torch.sqrt(x * x + y * y).clamp_min(1e-9)
    pitch = torch.atan2(z, horiz)                      # (B,N)
    yaw = torch.atan2(y, x)                            # (B,N) in [-pi,pi]

    # row = nearest beam angle
    row = (pitch.unsqueeze(-1) - row_angles).abs().argmin(dim=-1)  # (B,N)
    # col = uniform azimuth bin
    span = (fov_end - fov_start)
    col = ((yaw - fov_start) / span * W).floor().long().clamp(0, W - 1)  # (B,N)

    range_img = torch.full((B, H, W), float("inf"), device=dev)
    flat = (row * W + col)                                          # (B,N)
    for b in range(B):                                             # B is small
        ri = range_img[b].view(-1)
        ri.scatter_reduce_(0, flat[b], rng[b], reduce="amin", include_self=True)
    valid = torch.isfinite(range_img)
    range_img = torch.where(valid, range_img, torch.zeros_like(range_img))
    return range_img, valid, row, col


# --------------------------------------------------------------------------
# ground removal (vectorized angle-image method)
# --------------------------------------------------------------------------

def remove_ground(range_img, valid, row_angles, ground_angle_thresh=math.radians(45.0),
                  start_row=0):
    """Mark ground pixels via the vertical incline-angle method.

    Returns a (B,H,W) bool ground mask. Pixels whose incline angle to the row
    below is below `ground_angle_thresh` are flagged as ground. This is a
    vectorized approximation of DepthGroundRemover (no Savitzky-Golay smoothing).
    """
    B, H, W = range_img.shape
    ra = row_angles.to(range_img.device, torch.float32)
    d_cur = range_img[:, :-1, :]
    d_below = range_img[:, 1:, :]
    both = valid[:, :-1, :] & valid[:, 1:, :]
    dalpha = (ra[1:] - ra[:-1]).abs().view(1, H - 1, 1)
    # incline angle of the line between the two vertical beam hits
    dz = (d_below * torch.sin(ra[1:].view(1, -1, 1)) -
          d_cur * torch.sin(ra[:-1].view(1, -1, 1)))
    dx = (d_below * torch.cos(ra[1:].view(1, -1, 1)) -
          d_cur * torch.cos(ra[:-1].view(1, -1, 1)))
    angle = torch.atan2(dz.abs(), dx.abs().clamp_min(1e-9))
    ground = torch.zeros(B, H, W, dtype=torch.bool, device=range_img.device)
    is_g = both & (angle < ground_angle_thresh)
    ground[:, :-1, :] |= is_g
    ground[:, 1:, :] |= is_g
    return ground


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------

def cluster(range_img, valid, row_alphas, col_alphas, threshold,
            wrap=True, relabel=True, min_size=0, per_image_ids=False):
    """Cluster a batch of range images. Returns int labels (B,H,W).

    Background / invalid pixels get -1. If relabel=True, component ids are made
    consecutive per batch item starting at 0; otherwise global root indices are
    returned. min_size>0 drops components with fewer pixels (set to -1).
    """
    ext = _load_ext()
    valid = valid & (range_img > 0)
    labels = ext.cluster(range_img.float().contiguous(),
                          valid.contiguous(),
                          row_alphas, col_alphas,
                          float(threshold), bool(wrap))
    if min_size > 0:
        labels = _drop_small(labels, min_size)
    if relabel:
        labels = _relabel_consecutive(labels, per_image=per_image_ids)
    return labels


def _relabel_consecutive(labels, per_image=False):
    """Map component root indices to small consecutive ids, -1 stays background.

    Roots are global flat indices, hence already distinct across the batch, so
    by default a single GPU ``unique`` relabels the whole batch (ids 0..K-1 are
    globally unique). per_image=True restarts ids at 0 for each batch item
    (slower: one ``unique`` per image)."""
    out = torch.full_like(labels, -1)
    if per_image:
        for b in range(labels.shape[0]):
            lb = labels[b]
            mask = lb >= 0
            if not mask.any():
                continue
            _, inv = torch.unique(lb[mask], return_inverse=True)
            out[b][mask] = inv.to(out.dtype)
        return out
    mask = labels >= 0
    if mask.any():
        _, inv = torch.unique(labels[mask], return_inverse=True)
        out[mask] = inv.to(out.dtype)
    return out


def _drop_small(labels, min_size):
    for b in range(labels.shape[0]):
        lb = labels[b]
        mask = lb >= 0
        if not mask.any():
            continue
        uniq, counts = torch.unique(lb[mask], return_counts=True)
        small = uniq[counts < min_size]
        if small.numel():
            drop = torch.isin(lb, small)
            lb[drop] = -1
    return labels


# --------------------------------------------------------------------------
# pure-torch reference (label propagation) -- correctness oracle, not for prod
# --------------------------------------------------------------------------

def _beta(alpha, da, db):
    d1 = torch.maximum(da, db)
    d2 = torch.minimum(da, db)
    return torch.atan2(d2 * torch.sin(alpha), d1 - d2 * torch.cos(alpha)).abs()


def cluster_reference(range_img, valid, row_alphas, col_alphas, threshold,
                      wrap=True, max_iter=10000, relabel=True):
    """Slow but simple connected components via iterative label propagation.
    Used to validate the CUDA kernel."""
    B, H, W = range_img.shape
    dev = range_img.device
    valid = valid & (range_img > 0)
    ra = row_alphas.to(dev).view(1, H, 1)
    ca = col_alphas.to(dev).view(1, 1, W)

    idx = torch.arange(B * H * W, device=dev).view(B, H, W).int()
    lab = torch.where(valid, idx, torch.full_like(idx, -1))

    # edge masks (down, right) computed once
    down_ok = valid[:, :-1, :] & valid[:, 1:, :]
    down_ok &= _beta(ra[:, :-1, :], range_img[:, :-1, :], range_img[:, 1:, :]) > threshold
    rcur, rnext = range_img, torch.roll(range_img, shifts=-1, dims=2)
    right_ok = valid & torch.roll(valid, shifts=-1, dims=2)
    right_ok &= _beta(ca, rcur, rnext) > threshold
    if not wrap:
        right_ok[:, :, -1] = False

    BIG = torch.iinfo(torch.int32).max
    for _ in range(max_iter):
        cur = lab.clone()
        m = torch.where(lab >= 0, lab, torch.full_like(lab, BIG))
        nbr = torch.full_like(m, BIG)
        # down/up
        nbr[:, :-1, :] = torch.where(down_ok, torch.minimum(nbr[:, :-1, :], m[:, 1:, :]), nbr[:, :-1, :])
        nbr[:, 1:, :] = torch.where(down_ok, torch.minimum(nbr[:, 1:, :], m[:, :-1, :]), nbr[:, 1:, :])
        # right/left (with wrap via roll)
        mr = torch.roll(m, shifts=-1, dims=2)
        ml = torch.roll(m, shifts=1, dims=2)
        rol = torch.roll(right_ok, shifts=1, dims=2)
        nbr = torch.where(right_ok, torch.minimum(nbr, mr), nbr)
        nbr = torch.where(rol, torch.minimum(nbr, ml), nbr)
        newm = torch.minimum(m, nbr)
        lab = torch.where((lab >= 0) & (newm < BIG), newm, lab)
        if torch.equal(lab, cur):
            break
    if relabel:
        lab = _relabel_consecutive(lab)
    return lab
