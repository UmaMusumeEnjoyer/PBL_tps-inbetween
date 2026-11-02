# -*- coding: utf-8 -*-
"""
tps_anime/tps_solver.py
Hệ thống giải Thin-Plate Spline (TPS), 8 điểm neo biên/góc và biến dạng lưới tensor.
Tương thích hoàn toàn trên cả CPU và GPU.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_NORM_GRID_CACHE = {}
_PIXEL_GRID_CACHE = {}
_GRID_CACHE = {}


def warp_normalized_flow(img, flow):
    """
    Biến dạng lưới Coarse TPS (tọa độ chuẩn hóa [0, 1] sang [-1, 1]).
    Caching lưới cơ sở (1, H, W, 2) theo (h, w, device).
    """
    b, _, h, w = img.shape
    dev = flow.device
    key = (h, w, dev)
    base_grid = _NORM_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=dev, dtype=torch.float32),
            torch.linspace(0, 1, w, device=dev, dtype=torch.float32),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
        _NORM_GRID_CACHE[key] = base_grid

    flow_f32 = flow.float()
    grid = ((base_grid + flow_f32.permute(0, 2, 3, 1)) * 2.0 - 1.0).contiguous()
    out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)


def warp_pixel_flow(img, flow):
    """
    Biến dạng lưới Fine Optical Flow IFNet (tọa độ pixel thực tế).
    """
    b, _, h, w = img.shape
    dev = flow.device
    key = (h, w, dev)
    base_grid = _PIXEL_GRID_CACHE.get(key)
    if base_grid is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=dev, dtype=torch.float32),
            torch.linspace(-1, 1, w, device=dev, dtype=torch.float32),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()
        _PIXEL_GRID_CACHE[key] = base_grid

    flow_f32 = flow.float()
    u = flow_f32[:, 0, :, :] / ((w - 1.0) * 0.5)
    v = flow_f32[:, 1, :, :] / ((h - 1.0) * 0.5)
    flow_norm = torch.stack([u, v], dim=-1).contiguous()
    grid = (base_grid + flow_norm).contiguous()
    out = F.grid_sample(img.float().contiguous(), grid, padding_mode='border', align_corners=True)
    return out.to(img.dtype)


def tps_rbf(r):
    """Radial Basis Function: U(r) = r^2 * log(r + eps)"""
    return (r ** 2) * torch.log(torch.clamp(r, min=1e-6))


def filter_outlier_keypoints_cpu(c_s, c_d, max_pts=80, min_dist=0.015):
    """
    Lọc keypoints ngoại lai trên CPU bằng Numpy:
    1. Lọc displacement đột biến > 3.5 MAD.
    2. Spatial NMS: Tránh cụm điểm dính sát nhau làm ma trận K suy biến.
    """
    n = len(c_s)
    if n < 4:
        return c_s, c_d

    disp = c_d - c_s
    dists = np.linalg.norm(disp, axis=1)
    median_d = np.median(dists)
    mad = np.median(np.abs(dists - median_d)) + 1e-6
    mask_inliers = dists <= (median_d + 3.5 * mad)

    c_s_filt = c_s[mask_inliers]
    c_d_filt = c_d[mask_inliers]
    if len(c_s_filt) < 4:
        c_s_filt = c_s
        c_d_filt = c_d

    selected_s = []
    selected_d = []
    min_dist_sq = min_dist ** 2
    for p_s, p_d in zip(c_s_filt, c_d_filt):
        if len(selected_s) >= max_pts:
            break
        too_close = False
        for prev in selected_s:
            if ((p_s[0] - prev[0]) ** 2 + (p_s[1] - prev[1]) ** 2) < min_dist_sq:
                too_close = True
                break
        if not too_close:
            selected_s.append(p_s)
            selected_d.append(p_d)

    return np.array(selected_s, dtype=np.float32), np.array(selected_d, dtype=np.float32)


def solve_tps_single_cpu(c_src, c_dst, max_kps=80, lambd=0.5, min_dist=0.015, max_pts=88, return_tensor=True):
    """
    Giải hệ phương trình TPS cho 1 cặp ảnh với 8 ĐIỂM NEO BIÊN/GÓC:
    Đảm bảo 4 góc và 4 trung điểm viền luôn cố định (zero displacement).
    """
    if c_src is None or c_dst is None or len(c_src) < 4:
        z_c = np.zeros((max_pts, 2), dtype=np.float32)
        z_th = np.zeros((max_pts + 3, 2), dtype=np.float32)
        if return_tensor:
            return torch.from_numpy(z_c), torch.from_numpy(z_th), 0.0
        return z_c, z_th, 0.0

    c_s_clean, c_d_clean = filter_outlier_keypoints_cpu(c_src, c_dst, max_pts=max_kps, min_dist=min_dist)
    if len(c_s_clean) < 4:
        z_c = np.zeros((max_pts, 2), dtype=np.float32)
        z_th = np.zeros((max_pts + 3, 2), dtype=np.float32)
        if return_tensor:
            return torch.from_numpy(z_c), torch.from_numpy(z_th), 0.0
        return z_c, z_th, 0.0

    # 8 Điểm neo biên/góc bất biến tọa độ [0, 1]
    anchors = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5],             [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
    ], dtype=np.float32)

    c_s_anchors = np.vstack([c_s_clean, anchors])
    c_d_anchors = np.vstack([c_d_clean, anchors])
    m = len(c_s_anchors)

    diff = c_s_anchors[:, None, :] - c_s_anchors[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    K = (r ** 2) * np.log(np.maximum(r, 1e-6))
    np.fill_diagonal(K, lambd)

    P = np.hstack([np.ones((m, 1), dtype=np.float32), c_s_anchors])

    A = np.zeros((m + 3, m + 3), dtype=np.float32)
    A[:m, :m] = K
    A[:m, m:] = P
    A[m:, :m] = P.T

    Y = np.zeros((m + 3, 2), dtype=np.float32)
    Y[:m, :] = c_d_anchors

    try:
        theta = np.linalg.solve(A, Y)
    except np.linalg.LinAlgError:
        theta, _, _, _ = np.linalg.lstsq(A, Y, rcond=1e-4)

    v_flag = 0.0 if (np.isnan(theta).any() or np.isinf(theta).any() or np.max(np.abs(theta)) > 50.0) else 1.0

    c_pad = np.zeros((max_pts, 2), dtype=np.float32)
    theta_pad = np.zeros((max_pts + 3, 2), dtype=np.float32)
    if v_flag > 0.5:
        c_pad[:m] = c_s_anchors
        theta_pad[:m] = theta[:m]
        theta_pad[max_pts:max_pts + 3] = theta[m:m + 3]

    if return_tensor:
        return torch.from_numpy(c_pad), torch.from_numpy(theta_pad), float(v_flag)
    return c_pad, theta_pad, float(v_flag)


def solve_tps_pair_cpu(kps, max_kps=80, lambd=0.5, min_dist=0.015, return_tensor=True):
    """
    Giải 2 chiều (Forward 0->1 và Backward 1->0) cho 1 cặp keypoints.
    """
    MAX_PTS = 88
    if kps is None or len(kps) < 2:
        z_c0 = np.zeros((MAX_PTS, 2), dtype=np.float32)
        z_th01 = np.zeros((MAX_PTS + 3, 2), dtype=np.float32)
        z_c1 = np.zeros((MAX_PTS, 2), dtype=np.float32)
        z_th10 = np.zeros((MAX_PTS + 3, 2), dtype=np.float32)
        z_v = np.array(0.0, dtype=np.float32)
        if return_tensor:
            return (torch.from_numpy(z_c0), torch.from_numpy(z_th01), torch.from_numpy(z_c1),
                    torch.from_numpy(z_th10), torch.from_numpy(z_v))
        return z_c0, z_th01, z_c1, z_th10, z_v

    c0 = kps[0]
    c1 = kps[1]
    c0_pad, th01_pad, v01 = solve_tps_single_cpu(c0, c1, max_kps=max_kps, lambd=lambd, min_dist=min_dist, max_pts=MAX_PTS, return_tensor=False)
    c1_pad, th10_pad, v10 = solve_tps_single_cpu(c1, c0, max_kps=max_kps, lambd=lambd, min_dist=min_dist, max_pts=MAX_PTS, return_tensor=False)

    v = np.array(1.0 if (v01 > 0.5 and v10 > 0.5) else 0.0, dtype=np.float32)
    if return_tensor:
        return (torch.from_numpy(c0_pad), torch.from_numpy(th01_pad), torch.from_numpy(c1_pad),
                torch.from_numpy(th10_pad), torch.from_numpy(v))
    return c0_pad, th01_pad, c1_pad, th10_pad, v


def get_grid_pts(h, w, device):
    key = (h, w, device)
    pts = _GRID_CACHE.get(key)
    if pts is None:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=torch.float32),
            torch.linspace(0, 1, w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pts = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).contiguous()
        _GRID_CACHE[key] = pts
    return pts


@torch.no_grad()
def tps_flow_from_theta_batch(theta_batch, c_src_batch, valid_mask, h, w, device, downscale_factor=4):
    """
    Tính TPS Flow trên GPU theo cơ chế Hierarchical Fast Grid (mặc định downscale 4x sau đó bilinear upsample).
    """
    theta_f32 = torch.nan_to_num(theta_batch.float(), nan=0.0, posinf=0.0, neginf=0.0)
    c_src_f32 = torch.nan_to_num(c_src_batch.float(), nan=0.0, posinf=0.0, neginf=0.0)
    valid_f32 = torch.nan_to_num(valid_mask.float(), nan=0.0, posinf=0.0, neginf=0.0)
    b = theta_f32.shape[0]

    if downscale_factor > 1:
        h_sub = h // downscale_factor
        w_sub = w // downscale_factor
    else:
        h_sub, w_sub = h, w

    pts = get_grid_pts(h_sub, w_sub, device)
    total_pts = pts.shape[0]
    pts_exp = pts.unsqueeze(0).expand(b, -1, -1)

    dist = torch.cdist(pts_exp, c_src_f32, p=2.0)
    U = tps_rbf(dist)

    ones = torch.ones((b, total_pts, 1), device=device, dtype=pts.dtype)
    basis = torch.cat([U, ones, pts_exp], dim=-1)

    mapped = torch.bmm(basis, theta_f32)
    flow = (mapped - pts_exp).view(b, h_sub, w_sub, 2).permute(0, 3, 1, 2).contiguous()

    if downscale_factor > 1:
        flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=True)

    flow = flow * valid_f32.view(b, 1, 1, 1)
    return flow
