#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_fixedK_rt_depth_nn_gop.py

Stage-wise VGGT-Omega geometry canonicalization for codec input.

Recommended pipeline implemented here:
  Stage 1: fix one RAP-level K, keep original VGGT depth, optimize absolute R|t
           over codec-relevant adjacent + hierarchical/GOP anchor pairs.
  Stage 2: keep Stage-1 R fixed, optimize t plus a constrained tiny-NN
           multiplicative depth correction field.
  Stage 3: small-LR joint fine-tune of R/t/NN depth correction with priors
           toward the Stage-2 solution.

Depth correction is not an unconstrained depth generator. It is:
    depth'(x,y) = depth(x,y) * exp(g(x,y))
where g is a low-resolution correction field produced by a tiny CNN and clamped by
    g = max_log_scale * tanh(raw_g)
The final decoder does NOT need the NN. The script writes the final corrected depth
as YUV420p10le and writes fixed-K camera JSONL.

Input NPZ is expected from the user's VGGT-Omega runner:
  depth_original      [N,H,W] float32
  extrinsic           [N,3,4] camera_from_world [R|t]
  intrinsic_original  [N,3,3]
  frame_indices       optional [N]

Pair direction is target -> reference. Backward projection uses target-frame depth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Logging / misc
# ============================================================

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sanitize_windows_filename_component(name: str, replacement: str = "_") -> str:
    name = unicodedata.normalize("NFC", str(name))
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, name)
    if replacement:
        name = re.sub(re.escape(replacement) + r"+", replacement, name)
    name = name.rstrip(" .")
    if not name:
        name = "unnamed"
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if name.split(".")[0].upper() in reserved:
        name = "_" + name
    return name


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def as_float_list(a: np.ndarray | torch.Tensor) -> list[float]:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return [float(x) for x in np.asarray(a).reshape(-1)]


# ============================================================
# Camera math, numpy
# ============================================================

def split_extrinsic(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    E = np.asarray(E, dtype=np.float64)
    if E.shape != (3, 4):
        raise ValueError(f"extrinsic must be [3,4], got {E.shape}")
    return E[:, :3], E[:, 3]


def relative_current_to_ref(E_cur: np.ndarray, E_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given camera_from_world extrinsics, return X_ref = R_rel X_cur + t_rel."""
    R_cur, t_cur = split_extrinsic(E_cur)
    R_ref, t_ref = split_extrinsic(E_ref)
    R_rel = R_ref @ R_cur.T
    t_rel = t_ref - R_rel @ t_cur
    return R_rel, t_rel


def closest_rotation(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def rvec_from_R(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.reshape(3).astype(np.float64)


def R_from_rvec_np(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R.astype(np.float64)


def make_fixed_intrinsic(Ks: np.ndarray, width: int, height: int, center_mode: str) -> np.ndarray:
    Ks = np.asarray(Ks, dtype=np.float64)
    K0 = np.eye(3, dtype=np.float64)
    K0[0, 0] = float(np.median(Ks[:, 0, 0]))
    K0[1, 1] = float(np.median(Ks[:, 1, 1]))
    if center_mode == "image-center":
        K0[0, 2] = float(width) / 2.0
        K0[1, 2] = float(height) / 2.0
    elif center_mode == "median":
        K0[0, 2] = float(np.median(Ks[:, 0, 2]))
        K0[1, 2] = float(np.median(Ks[:, 1, 2]))
    elif center_mode == "first":
        K0[0, 2] = float(Ks[0, 0, 2])
        K0[1, 2] = float(Ks[0, 1, 2])
    else:
        raise ValueError(center_mode)
    return K0


def make_rays_np(K: np.ndarray, width: int, height: int, z_sign: float = 1.0) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays = np.empty((height, width, 3), dtype=np.float64)
    rays[..., 0] = (xs - cx) / fx
    rays[..., 1] = (ys - cy) / fy
    rays[..., 2] = float(z_sign)
    return rays


def project_points_np(X: np.ndarray, K: np.ndarray, z_sign: float, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = X[..., 2]
    valid = np.isfinite(X).all(axis=-1) & (Z * z_sign > eps)
    denom = np.where(np.abs(Z) > eps, Z, np.where(Z >= 0, eps, -eps))
    mx = float(K[0, 0]) * (X[..., 0] / denom) + float(K[0, 2])
    my = float(K[1, 1]) * (X[..., 1] / denom) + float(K[1, 2])
    return mx.astype(np.float32), my.astype(np.float32), valid


# ============================================================
# Torch SO(3) / projection
# ============================================================

def rodrigues_torch(rvec: torch.Tensor) -> torch.Tensor:
    """Batched Rodrigues. rvec: [N,3], returns [N,3,3]."""
    dtype = rvec.dtype
    device = rvec.device
    N = rvec.shape[0]
    x, y, z = rvec[:, 0], rvec[:, 1], rvec[:, 2]
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z, y], dim=-1),
        torch.stack([z, zero, -x], dim=-1),
        torch.stack([-y, x, zero], dim=-1),
    ], dim=-2)
    theta2 = torch.sum(rvec * rvec, dim=-1)
    theta = torch.sqrt(torch.clamp(theta2, min=1e-30))
    small = theta2 < 1e-12
    A = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, torch.sin(theta) / theta)
    B = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - torch.cos(theta)) / theta2)
    I = torch.eye(3, dtype=dtype, device=device).expand(N, 3, 3)
    R = I + A[:, None, None] * K + B[:, None, None] * (K @ K)
    return R


def robust_epe_loss(pred_xy: torch.Tensor, target_xy: torch.Tensor, f_scale: float) -> torch.Tensor:
    diff = pred_xy - target_xy
    e2 = torch.sum(diff * diff, dim=-1)
    fs = float(f_scale)
    return torch.mean(torch.sqrt(e2 + fs * fs) - fs)


def project_samples_torch(
    rays: torch.Tensor,
    depth: torch.Tensor,
    R_rel: torch.Tensor,
    t_rel: torch.Tensor,
    K_fixed_t: torch.Tensor,
    z_sign: float,
    depth_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if depth_scale is not None:
        depth = depth * depth_scale
    X_cur = depth[:, None] * rays
    X_ref = X_cur @ R_rel.T + t_rel[None, :]
    Z = X_ref[:, 2]
    eps = torch.tensor(1e-8, dtype=X_ref.dtype, device=X_ref.device)
    denom = torch.where(torch.abs(Z) > eps, Z, torch.where(Z >= 0, eps, -eps))
    mx = K_fixed_t[0, 0] * (X_ref[:, 0] / denom) + K_fixed_t[0, 2]
    my = K_fixed_t[1, 1] * (X_ref[:, 1] / denom) + K_fixed_t[1, 2]
    return torch.stack([mx, my], dim=-1)


# ============================================================
# Pair generation and sampling
# ============================================================

@dataclass(frozen=True)
class PairSpec:
    target: int
    ref: int
    weight: float
    kind: str


def add_pair_accum(acc: dict[tuple[int, int], PairSpec], target: int, ref: int, weight: float, kind: str, n: int) -> None:
    if target == ref:
        return
    if not (0 <= target < n and 0 <= ref < n):
        return
    key = (int(target), int(ref))
    if key in acc:
        old = acc[key]
        acc[key] = PairSpec(key[0], key[1], old.weight + float(weight), old.kind + "+" + kind)
    else:
        acc[key] = PairSpec(key[0], key[1], float(weight), kind)


def generate_hierarchical_pairs(n: int, bidirectional: bool, weight: float) -> list[PairSpec]:
    acc: dict[tuple[int, int], PairSpec] = {}

    def rec(a: int, b: int, level: int) -> None:
        if b - a < 2:
            return
        mid = (a + b) // 2
        if mid == a or mid == b:
            return
        w = float(weight) / math.sqrt(level + 1.0)
        # Codec-current direction: mid is coded from anchors a/b.
        add_pair_accum(acc, mid, a, w, f"gop_L{level}", n)
        add_pair_accum(acc, mid, b, w, f"gop_L{level}", n)
        if bidirectional:
            # Reverse consistency, useful for tool propagation / symmetric reference use.
            add_pair_accum(acc, a, mid, w, f"gop_rev_L{level}", n)
            add_pair_accum(acc, b, mid, w, f"gop_rev_L{level}", n)
        rec(a, mid, level + 1)
        rec(mid, b, level + 1)

    rec(0, n - 1, 0)
    return list(acc.values())


def parse_extra_pairs(s: str | None, n: int) -> list[PairSpec]:
    if not s:
        return []
    out: list[PairSpec] = []
    for tok in re.split(r"[,;\s]+", s.strip()):
        if not tok:
            continue
        tok = tok.replace("->", ":")
        parts = tok.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Bad pair token: {tok}. Use target:ref[:weight].")
        target = int(parts[0])
        ref = int(parts[1])
        weight = float(parts[2]) if len(parts) == 3 else 1.0
        if not (0 <= target < n and 0 <= ref < n):
            raise ValueError(f"Pair out of range for N={n}: {tok}")
        out.append(PairSpec(target, ref, weight, "extra"))
    return out


def build_pair_list(
    n: int,
    include_adjacent: bool,
    adjacent_bidirectional: bool,
    adjacent_weight: float,
    include_gop: bool,
    bidirectional_gop: bool,
    gop_weight: float,
    extra_pairs: str | None,
) -> list[PairSpec]:
    acc: dict[tuple[int, int], PairSpec] = {}
    if include_adjacent:
        for i in range(1, n):
            add_pair_accum(acc, i, i - 1, adjacent_weight, "adjacent", n)
            if adjacent_bidirectional:
                add_pair_accum(acc, i - 1, i, adjacent_weight, "adjacent_rev", n)
    if include_gop:
        for p in generate_hierarchical_pairs(n, bidirectional=bidirectional_gop, weight=gop_weight):
            add_pair_accum(acc, p.target, p.ref, p.weight, p.kind, n)
    for p in parse_extra_pairs(extra_pairs, n):
        add_pair_accum(acc, p.target, p.ref, p.weight, p.kind, n)
    return sorted(acc.values(), key=lambda p: (abs(p.target - p.ref), p.target, p.ref))


@dataclass
class PairCacheNP:
    spec: PairSpec
    y: np.ndarray
    x: np.ndarray
    rays: np.ndarray
    depth: np.ndarray
    target_xy: np.ndarray


@dataclass
class PairCacheTorch:
    spec: PairSpec
    y: torch.Tensor
    x: torch.Tensor
    rays: torch.Tensor
    depth: torch.Tensor
    target_xy: torch.Tensor


def create_pair_cache(
    pairs: list[PairSpec],
    depth: np.ndarray,
    E_abs: np.ndarray,
    K_orig: np.ndarray,
    K_fixed: np.ndarray,
    rays_fixed_full: np.ndarray,
    sample_stride: int,
    max_samples_per_pair: int,
    z_sign: float,
    seed: int,
) -> list[PairCacheNP]:
    rng = np.random.default_rng(seed)
    n, h, w = depth.shape
    inv_K_fixed = np.linalg.inv(K_fixed)
    H = np.stack([inv_K_fixed @ K_orig[i] for i in range(n)], axis=0)
    H_inv = np.stack([np.linalg.inv(H[i]) for i in range(n)], axis=0)

    yy, xx = np.meshgrid(np.arange(0, h, sample_stride), np.arange(0, w, sample_stride), indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    out: list[PairCacheNP] = []

    for idx, p in enumerate(pairs):
        tar, ref = p.target, p.ref
        R_rel_raw, t_rel_raw = relative_current_to_ref(E_abs[tar], E_abs[ref])
        # Exact target in fixed-K coordinates if affine cameras were allowed.
        A_exact = H[ref] @ R_rel_raw @ H_inv[tar]
        b_exact = H[ref] @ t_rel_raw

        d = depth[tar, yy, xx].astype(np.float64)
        rays = rays_fixed_full[yy, xx].astype(np.float64)
        valid_depth = np.isfinite(d) & (d > 0)
        X_cur = d[:, None] * rays
        X_ref = X_cur @ A_exact.T + b_exact.reshape(1, 3)
        mx, my, valid_z = project_points_np(X_ref, K_fixed, z_sign=z_sign)
        valid = valid_depth & valid_z & np.isfinite(mx) & np.isfinite(my) & (mx >= 0.0) & (mx <= w - 1) & (my >= 0.0) & (my <= h - 1)
        ids = np.flatnonzero(valid)
        if ids.size == 0:
            log(f"WARNING: pair {tar}->{ref} has no valid samples; skipped")
            continue
        if max_samples_per_pair > 0 and ids.size > max_samples_per_pair:
            ids = rng.choice(ids, size=max_samples_per_pair, replace=False)
            ids.sort()
        y_sel = yy[ids].astype(np.int64)
        x_sel = xx[ids].astype(np.int64)
        out.append(PairCacheNP(
            spec=p,
            y=y_sel,
            x=x_sel,
            rays=rays[ids].astype(np.float32),
            depth=d[ids].astype(np.float32),
            target_xy=np.stack([mx[ids], my[ids]], axis=1).astype(np.float32),
        ))
        log(f"Cached pair {idx+1:03d}/{len(pairs):03d}: {tar}->{ref}, weight={p.weight:.3g}, samples={len(ids)}, kind={p.kind}")
    return out


def cache_to_torch(cache_np: list[PairCacheNP], device: torch.device, dtype: torch.dtype) -> list[PairCacheTorch]:
    out: list[PairCacheTorch] = []
    for c in cache_np:
        out.append(PairCacheTorch(
            spec=c.spec,
            y=torch.from_numpy(c.y).to(device=device, dtype=torch.long),
            x=torch.from_numpy(c.x).to(device=device, dtype=torch.long),
            rays=torch.from_numpy(c.rays).to(device=device, dtype=dtype),
            depth=torch.from_numpy(c.depth).to(device=device, dtype=dtype),
            target_xy=torch.from_numpy(c.target_xy).to(device=device, dtype=dtype),
        ))
    return out


# ============================================================
# NN depth correction
# ============================================================

class DepthCorrectionNet(nn.Module):
    """
    Tiny per-RAP low-res NN correction field.

    Input to CNN is concat([fixed depth-derived features, learnable latent]).
    Output raw_g is converted to bounded log-depth-scale by:
        g = max_log_scale * tanh(raw_g)
    """
    def __init__(self, feature_ch: int, latent_ch: int = 2, hidden_ch: int = 24, layers: int = 3):
        super().__init__()
        self.latent_ch = int(latent_ch)
        self.hidden_ch = int(hidden_ch)
        in_ch = feature_ch + latent_ch
        mods: list[nn.Module] = []
        ch = in_ch
        for _ in range(max(1, layers - 1)):
            mods.append(nn.Conv2d(ch, hidden_ch, kernel_size=3, padding=1))
            mods.append(nn.SiLU(inplace=True))
            ch = hidden_ch
        mods.append(nn.Conv2d(ch, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*mods)
        # Start from identity depth correction.
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor, latent: torch.Tensor, max_log_scale: float) -> torch.Tensor:
        raw = self.net(torch.cat([features, latent], dim=1))
        return float(max_log_scale) * torch.tanh(raw)


def build_depth_nn_features_np(depth: np.ndarray, grid_cell: int, include_xy: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return features [N,C,Gh,Gw], edge weights wx [N,1,Gh,Gw-1], wy [N,1,Gh-1,Gw]."""
    n, h, w = depth.shape
    gh = int(math.ceil(h / grid_cell))
    gw = int(math.ceil(w / grid_cell))
    logd = np.log(np.maximum(depth.astype(np.float32), 1e-12))
    log_low = np.empty((n, gh, gw), dtype=np.float32)
    for i in range(n):
        # AREA is stable for downsampling; output size is (gw, gh).
        log_low[i] = cv2.resize(logd[i], (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
    mean = float(np.mean(log_low[np.isfinite(log_low)]))
    std = float(np.std(log_low[np.isfinite(log_low)]))
    std = max(std, 1e-6)
    log_norm = (log_low - mean) / std

    dx = np.zeros_like(log_norm)
    dy = np.zeros_like(log_norm)
    if gw > 1:
        dx[:, :, 1:] = np.abs(log_low[:, :, 1:] - log_low[:, :, :-1])
        dx[:, :, 0] = dx[:, :, 1]
    if gh > 1:
        dy[:, 1:, :] = np.abs(log_low[:, 1:, :] - log_low[:, :-1, :])
        dy[:, 0, :] = dy[:, 1, :]
    grad = np.sqrt(dx * dx + dy * dy)
    gstd = float(np.std(grad[np.isfinite(grad)]))
    gstd = max(gstd, 1e-6)
    grad_norm = grad / gstd

    feats = [log_norm[:, None, :, :].astype(np.float32), grad_norm[:, None, :, :].astype(np.float32)]
    if include_xy:
        xs = np.linspace(-1.0, 1.0, gw, dtype=np.float32)[None, None, None, :]
        ys = np.linspace(-1.0, 1.0, gh, dtype=np.float32)[None, None, :, None]
        xx = np.broadcast_to(xs, (n, 1, gh, gw)).copy()
        yy = np.broadcast_to(ys, (n, 1, gh, gw)).copy()
        feats += [xx, yy]
    # frame index channel helps per-frame variation while preserving CNN structure.
    if n > 1:
        ts = np.linspace(-1.0, 1.0, n, dtype=np.float32)[:, None, None, None]
    else:
        ts = np.zeros((n, 1, 1, 1), dtype=np.float32)
    tt = np.broadcast_to(ts, (n, 1, gh, gw)).copy()
    feats.append(tt)
    features = np.concatenate(feats, axis=1).astype(np.float32)

    # Edge-aware smoothness weights from low-res log-depth discontinuities.
    # Normalize by median nonzero diff for sequence robustness.
    if gw > 1:
        diffx = np.abs(log_low[:, :, 1:] - log_low[:, :, :-1])
    else:
        diffx = np.zeros((n, gh, 0), dtype=np.float32)
    if gh > 1:
        diffy = np.abs(log_low[:, 1:, :] - log_low[:, :-1, :])
    else:
        diffy = np.zeros((n, 0, gw), dtype=np.float32)
    vals = np.concatenate([diffx.reshape(-1), diffy.reshape(-1)]) if (diffx.size + diffy.size) else np.zeros((1,), dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    med = float(np.median(vals[vals > 0])) if np.any(vals > 0) else 1.0
    med = max(med, 1e-6)
    # Use a mild default scale here; CLI alpha multiplies this.
    wx = np.exp(-diffx / med).astype(np.float32)[:, None, :, :] if diffx.size else np.zeros((n, 1, gh, 0), dtype=np.float32)
    wy = np.exp(-diffy / med).astype(np.float32)[:, None, :, :] if diffy.size else np.zeros((n, 1, 0, gw), dtype=np.float32)
    return features, wx, wy


def sample_g_from_map(g_maps: torch.Tensor, target: int, x: torch.Tensor, y: torch.Tensor, width: int, height: int) -> torch.Tensor:
    # g_maps: [N,1,Gh,Gw]. grid_sample coords are normalized to [-1,1].
    gx = 2.0 * x.to(dtype=g_maps.dtype) / max(width - 1, 1) - 1.0
    gy = 2.0 * y.to(dtype=g_maps.dtype) / max(height - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    vals = F.grid_sample(g_maps[target:target + 1], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return vals.view(-1)


def depth_nn_regularization(
    g_maps: torch.Tensor,
    wx: torch.Tensor,
    wy: torch.Tensor,
    l2: float,
    spatial: float,
    temporal: float,
    edge_alpha: float,
) -> torch.Tensor:
    reg = torch.zeros((), dtype=g_maps.dtype, device=g_maps.device)
    if l2 > 0:
        reg = reg + float(l2) * torch.mean(g_maps * g_maps)
    if temporal > 0 and g_maps.shape[0] > 1:
        reg = reg + float(temporal) * torch.mean((g_maps[1:] - g_maps[:-1]) ** 2)
    if spatial > 0:
        if g_maps.shape[-1] > 1:
            dx = g_maps[:, :, :, 1:] - g_maps[:, :, :, :-1]
            # wx is in [0,1]; edge_alpha sharpens/softens it.
            wxx = torch.clamp(wx, min=1e-4, max=1.0) ** float(edge_alpha)
            reg = reg + float(spatial) * torch.mean(wxx * dx * dx)
        if g_maps.shape[-2] > 1:
            dy = g_maps[:, :, 1:, :] - g_maps[:, :, :-1, :]
            wyy = torch.clamp(wy, min=1e-4, max=1.0) ** float(edge_alpha)
            reg = reg + float(spatial) * torch.mean(wyy * dy * dy)
    return reg


def upsample_g_maps_np(g_maps: np.ndarray, height: int, width: int) -> np.ndarray:
    # g_maps [N,1,Gh,Gw] -> [N,H,W]
    n = g_maps.shape[0]
    out = np.empty((n, height, width), dtype=np.float32)
    for i in range(n):
        out[i] = cv2.resize(g_maps[i, 0].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return out


def bilinear_sample_g_np(g_map: np.ndarray, x: np.ndarray, y: np.ndarray, width: int, height: int) -> np.ndarray:
    # g_map is [Gh,Gw] and aligned to full image extent. Sample at full-res coords.
    gh, gw = g_map.shape
    if gw <= 1:
        gx = np.zeros_like(x, dtype=np.float64)
    else:
        gx = np.clip(x.astype(np.float64) * (gw - 1) / max(width - 1, 1), 0, gw - 1)
    if gh <= 1:
        gy = np.zeros_like(y, dtype=np.float64)
    else:
        gy = np.clip(y.astype(np.float64) * (gh - 1) / max(height - 1, 1), 0, gh - 1)
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    x1 = np.minimum(x0 + 1, gw - 1)
    y1 = np.minimum(y0 + 1, gh - 1)
    wx = gx - x0
    wy = gy - y0
    v00 = g_map[y0, x0]
    v10 = g_map[y0, x1]
    v01 = g_map[y1, x0]
    v11 = g_map[y1, x1]
    return ((1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v10 + (1 - wx) * wy * v01 + wx * wy * v11).astype(np.float32)


# ============================================================
# Optimization config
# ============================================================

@dataclass
class OptimConfig:
    device: str
    dtype: str
    seed: int
    fixed_center_mode: str
    z_sign: float
    include_adjacent: bool
    adjacent_bidirectional: bool
    adjacent_weight: float
    include_gop: bool
    bidirectional_gop: bool
    gop_weight: float
    extra_pairs: str | None
    sample_stride: int
    max_samples_per_pair: int
    stage1_iters: int
    stage1_lr_r: float
    stage1_lr_t: float
    stage1_rot_prior: float
    stage1_t_prior: float
    stage2_iters: int
    stage2_lr_t: float
    stage2_lr_nn: float
    stage2_t_prior: float
    stage3_iters: int
    stage3_lr_r: float
    stage3_lr_t: float
    stage3_lr_nn: float
    stage3_rot_prior: float
    stage3_t_prior: float
    stage3_depth_prior: float
    nn_grid_cell: int
    nn_hidden_ch: int
    nn_latent_ch: int
    nn_layers: int
    depth_max_log_scale: float
    depth_l2: float
    depth_spatial_smooth: float
    depth_temporal_smooth: float
    depth_edge_alpha: float
    f_scale: float
    print_every: int
    depth_scale_precision: int
    depth_scale_percentile: float


def init_fixedK_absolute_poses(E_abs: np.ndarray, K_orig: np.ndarray, K_fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate each fixed-K affine absolute camera H_i [R_i|t_i] by rigid [R0_i|t0_i].
    X_fixed_cam_i = H_i X_raw_cam_i = H_i R_i X_world + H_i t_i.
    """
    n = E_abs.shape[0]
    H = np.stack([np.linalg.inv(K_fixed) @ K_orig[i] for i in range(n)], axis=0)
    R0 = np.zeros((n, 3, 3), dtype=np.float64)
    t0 = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        R_i, t_i = split_extrinsic(E_abs[i])
        R0[i] = closest_rotation(H[i] @ R_i)
        t0[i] = H[i] @ t_i
    rvec0 = np.stack([rvec_from_R(R0[i]) for i in range(n)], axis=0)
    return rvec0, t0


def build_full_pose_tensors(
    r_free: torch.Tensor,
    t_free: torch.Tensor,
    r0_fixed: torch.Tensor,
    t0_fixed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r_all = torch.cat([r0_fixed[None, :], r_free], dim=0)
    t_all = torch.cat([t0_fixed[None, :], t_free], dim=0)
    R_all = rodrigues_torch(r_all)
    return r_all, t_all, R_all


def pair_loss_sum(
    pair_cache: list[PairCacheTorch],
    R_all: torch.Tensor,
    t_all: torch.Tensor,
    K_fixed_t: torch.Tensor,
    width: int,
    height: int,
    z_sign: float,
    f_scale: float,
    g_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    total = torch.zeros((), dtype=K_fixed_t.dtype, device=K_fixed_t.device)
    wsum = 0.0
    for c in pair_cache:
        tar, ref = c.spec.target, c.spec.ref
        R_tar = R_all[tar]
        R_ref = R_all[ref]
        t_tar = t_all[tar]
        t_ref = t_all[ref]
        R_rel = R_ref @ R_tar.T
        t_rel = t_ref - R_rel @ t_tar

        depth_scale = None
        if g_maps is not None:
            g = sample_g_from_map(g_maps, tar, c.x, c.y, width, height)
            depth_scale = torch.exp(g)

        pred = project_samples_torch(c.rays, c.depth, R_rel, t_rel, K_fixed_t, z_sign, depth_scale)
        loss = robust_epe_loss(pred, c.target_xy, f_scale=f_scale)
        w = float(c.spec.weight)
        total = total + w * loss
        wsum += w
    return total / max(wsum, 1e-12)


def optimize_stage1_rt(
    pair_cache: list[PairCacheTorch],
    r_init_np: np.ndarray,
    t_init_np: np.ndarray,
    K_fixed: np.ndarray,
    width: int,
    height: int,
    cfg: OptimConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    log("Stage 1: fixed K + original depth + optimize absolute R|t")
    r_init = torch.tensor(r_init_np, device=device, dtype=dtype)
    t_init = torch.tensor(t_init_np, device=device, dtype=dtype)
    K_fixed_t = torch.tensor(K_fixed, device=device, dtype=dtype)
    r0_fixed = r_init[0].detach()
    t0_fixed = t_init[0].detach()
    r_free = torch.nn.Parameter(r_init[1:].clone())
    t_free = torch.nn.Parameter(t_init[1:].clone())
    opt = torch.optim.Adam([
        {"params": [r_free], "lr": cfg.stage1_lr_r},
        {"params": [t_free], "lr": cfg.stage1_lr_t},
    ])
    hist: list[dict[str, float]] = []
    for it in range(1, cfg.stage1_iters + 1):
        opt.zero_grad(set_to_none=True)
        r_all, t_all, R_all = build_full_pose_tensors(r_free, t_free, r0_fixed, t0_fixed)
        loss_pair = pair_loss_sum(pair_cache, R_all, t_all, K_fixed_t, width, height, cfg.z_sign, cfg.f_scale)
        loss = loss_pair
        if cfg.stage1_rot_prior > 0:
            loss = loss + cfg.stage1_rot_prior * torch.mean((r_all - r_init) ** 2)
        if cfg.stage1_t_prior > 0:
            loss = loss + cfg.stage1_t_prior * torch.mean((t_all - t_init) ** 2)
        loss.backward()
        opt.step()
        if it == 1 or it == cfg.stage1_iters or (cfg.print_every > 0 and it % cfg.print_every == 0):
            rec = {"iter": it, "loss": float(loss.detach().cpu()), "pair_loss": float(loss_pair.detach().cpu())}
            hist.append(rec)
            log(f"Stage1 iter {it:04d}/{cfg.stage1_iters}: loss={rec['loss']:.6f}, pair={rec['pair_loss']:.6f}")
    with torch.no_grad():
        r_all, t_all, _ = build_full_pose_tensors(r_free, t_free, r0_fixed, t0_fixed)
    return r_all.detach().cpu().numpy(), t_all.detach().cpu().numpy(), hist


def optimize_stage2_t_nn(
    pair_cache: list[PairCacheTorch],
    r_stage1_np: np.ndarray,
    t_stage1_np: np.ndarray,
    K_fixed: np.ndarray,
    features: torch.Tensor,
    wx: torch.Tensor,
    wy: torch.Tensor,
    cfg: OptimConfig,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, DepthCorrectionNet, torch.Tensor, np.ndarray, list[dict[str, float]]]:
    log("Stage 2: fixed K + fixed R + optimize t and tiny-NN depth correction")
    n, feature_ch, gh, gw = features.shape
    r_stage1 = torch.tensor(r_stage1_np, device=device, dtype=dtype)
    t_stage1 = torch.tensor(t_stage1_np, device=device, dtype=dtype)
    K_fixed_t = torch.tensor(K_fixed, device=device, dtype=dtype)
    R_fixed = rodrigues_torch(r_stage1).detach()

    t0_fixed = t_stage1[0].detach()
    t_free = torch.nn.Parameter(t_stage1[1:].clone())
    net = DepthCorrectionNet(feature_ch=feature_ch, latent_ch=cfg.nn_latent_ch, hidden_ch=cfg.nn_hidden_ch, layers=cfg.nn_layers).to(device=device, dtype=dtype)
    latent = torch.nn.Parameter(torch.zeros((n, cfg.nn_latent_ch, gh, gw), dtype=dtype, device=device))

    opt = torch.optim.Adam([
        {"params": [t_free], "lr": cfg.stage2_lr_t},
        {"params": list(net.parameters()) + [latent], "lr": cfg.stage2_lr_nn},
    ])
    hist: list[dict[str, float]] = []
    for it in range(1, cfg.stage2_iters + 1):
        opt.zero_grad(set_to_none=True)
        t_all = torch.cat([t0_fixed[None, :], t_free], dim=0)
        g_maps = net(features, latent, cfg.depth_max_log_scale)
        loss_pair = pair_loss_sum(pair_cache, R_fixed, t_all, K_fixed_t, width, height, cfg.z_sign, cfg.f_scale, g_maps=g_maps)
        loss_reg = torch.zeros_like(loss_pair)
        if cfg.stage2_t_prior > 0:
            loss_reg = loss_reg + cfg.stage2_t_prior * torch.mean((t_all - t_stage1) ** 2)
        loss_reg = loss_reg + depth_nn_regularization(g_maps, wx, wy, cfg.depth_l2, cfg.depth_spatial_smooth, cfg.depth_temporal_smooth, cfg.depth_edge_alpha)
        loss = loss_pair + loss_reg
        loss.backward()
        opt.step()
        if it == 1 or it == cfg.stage2_iters or (cfg.print_every > 0 and it % cfg.print_every == 0):
            rec = {
                "iter": it,
                "loss": float(loss.detach().cpu()),
                "pair_loss": float(loss_pair.detach().cpu()),
                "reg_loss": float(loss_reg.detach().cpu()),
            }
            hist.append(rec)
            log(f"Stage2 iter {it:04d}/{cfg.stage2_iters}: loss={rec['loss']:.6f}, pair={rec['pair_loss']:.6f}, reg={rec['reg_loss']:.6f}")
    with torch.no_grad():
        t_all = torch.cat([t0_fixed[None, :], t_free], dim=0)
        g_maps = net(features, latent, cfg.depth_max_log_scale)
        g_np = g_maps.detach().cpu().numpy().astype(np.float32)
    return t_all.detach().cpu().numpy(), net, latent.detach().clone(), g_np, hist


def optimize_stage3_joint(
    pair_cache: list[PairCacheTorch],
    r_stage2_np: np.ndarray,
    t_stage2_np: np.ndarray,
    net: DepthCorrectionNet,
    latent_base: torch.Tensor,
    g_base_np: np.ndarray,
    K_fixed: np.ndarray,
    features: torch.Tensor,
    wx: torch.Tensor,
    wy: torch.Tensor,
    cfg: OptimConfig,
    width: int,
    height: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, DepthCorrectionNet, torch.Tensor, np.ndarray, list[dict[str, float]]]:
    log("Stage 3: small-LR joint fine-tuning of R/t/NN depth correction")
    r_base = torch.tensor(r_stage2_np, device=device, dtype=dtype)
    t_base = torch.tensor(t_stage2_np, device=device, dtype=dtype)
    g_base = torch.tensor(g_base_np, device=device, dtype=dtype)
    K_fixed_t = torch.tensor(K_fixed, device=device, dtype=dtype)

    r0_fixed = r_base[0].detach()
    t0_fixed = t_base[0].detach()
    r_free = torch.nn.Parameter(r_base[1:].clone())
    t_free = torch.nn.Parameter(t_base[1:].clone())
    latent = torch.nn.Parameter(latent_base.clone().detach())

    opt = torch.optim.Adam([
        {"params": [r_free], "lr": cfg.stage3_lr_r},
        {"params": [t_free], "lr": cfg.stage3_lr_t},
        {"params": list(net.parameters()) + [latent], "lr": cfg.stage3_lr_nn},
    ])
    hist: list[dict[str, float]] = []
    for it in range(1, cfg.stage3_iters + 1):
        opt.zero_grad(set_to_none=True)
        r_all, t_all, R_all = build_full_pose_tensors(r_free, t_free, r0_fixed, t0_fixed)
        g_maps = net(features, latent, cfg.depth_max_log_scale)
        loss_pair = pair_loss_sum(pair_cache, R_all, t_all, K_fixed_t, width, height, cfg.z_sign, cfg.f_scale, g_maps=g_maps)
        loss_reg = torch.zeros_like(loss_pair)
        if cfg.stage3_rot_prior > 0:
            loss_reg = loss_reg + cfg.stage3_rot_prior * torch.mean((r_all - r_base) ** 2)
        if cfg.stage3_t_prior > 0:
            loss_reg = loss_reg + cfg.stage3_t_prior * torch.mean((t_all - t_base) ** 2)
        loss_reg = loss_reg + depth_nn_regularization(g_maps, wx, wy, cfg.depth_l2, cfg.depth_spatial_smooth, cfg.depth_temporal_smooth, cfg.depth_edge_alpha)
        if cfg.stage3_depth_prior > 0:
            loss_reg = loss_reg + cfg.stage3_depth_prior * torch.mean((g_maps - g_base) ** 2)
        loss = loss_pair + loss_reg
        loss.backward()
        opt.step()
        if it == 1 or it == cfg.stage3_iters or (cfg.print_every > 0 and it % cfg.print_every == 0):
            rec = {
                "iter": it,
                "loss": float(loss.detach().cpu()),
                "pair_loss": float(loss_pair.detach().cpu()),
                "reg_loss": float(loss_reg.detach().cpu()),
            }
            hist.append(rec)
            log(f"Stage3 iter {it:04d}/{cfg.stage3_iters}: loss={rec['loss']:.6f}, pair={rec['pair_loss']:.6f}, reg={rec['reg_loss']:.6f}")
    with torch.no_grad():
        r_all, t_all, _ = build_full_pose_tensors(r_free, t_free, r0_fixed, t0_fixed)
        g_maps = net(features, latent, cfg.depth_max_log_scale)
        g_np = g_maps.detach().cpu().numpy().astype(np.float32)
    return r_all.detach().cpu().numpy(), t_all.detach().cpu().numpy(), net, latent.detach().clone(), g_np, hist


# ============================================================
# Evaluation / output
# ============================================================

def evaluate_on_cache(
    name: str,
    cache_np: list[PairCacheNP],
    r_np: np.ndarray,
    t_np: np.ndarray,
    K_fixed: np.ndarray,
    width: int,
    height: int,
    z_sign: float,
    g_maps: np.ndarray | None,
) -> dict[str, Any]:
    R_all = np.stack([R_from_rvec_np(r_np[i]) for i in range(len(r_np))], axis=0)
    fx, fy, cx, cy = float(K_fixed[0, 0]), float(K_fixed[1, 1]), float(K_fixed[0, 2]), float(K_fixed[1, 2])
    per_pair = []
    all_means = []
    all_p95 = []
    for c in cache_np:
        tar, ref = c.spec.target, c.spec.ref
        R_rel = R_all[ref] @ R_all[tar].T
        t_rel = t_np[ref] - R_rel @ t_np[tar]
        d = c.depth.astype(np.float64).copy()
        if g_maps is not None:
            g = bilinear_sample_g_np(g_maps[tar, 0], c.x, c.y, width, height)
            d *= np.exp(g.astype(np.float64))
        X = d[:, None] * c.rays.astype(np.float64)
        Xr = X @ R_rel.T + t_rel.reshape(1, 3)
        Z = Xr[:, 2]
        valid = np.isfinite(Xr).all(axis=1) & (Z * z_sign > 1e-8)
        denom = np.where(np.abs(Z) > 1e-8, Z, np.where(Z >= 0, 1e-8, -1e-8))
        mx = fx * (Xr[:, 0] / denom) + cx
        my = fy * (Xr[:, 1] / denom) + cy
        valid &= np.isfinite(mx) & np.isfinite(my)
        if np.count_nonzero(valid) == 0:
            rec = {"target": tar, "ref": ref, "kind": c.spec.kind, "weight": c.spec.weight, "valid_count": 0, "mean_epe": None, "p95_epe": None}
        else:
            dx = mx[valid] - c.target_xy[valid, 0]
            dy = my[valid] - c.target_xy[valid, 1]
            epe = np.sqrt(dx * dx + dy * dy)
            rec = {
                "target": tar,
                "ref": ref,
                "kind": c.spec.kind,
                "weight": float(c.spec.weight),
                "valid_count": int(np.count_nonzero(valid)),
                "mean_epe": float(np.mean(epe)),
                "p50_epe": float(np.percentile(epe, 50)),
                "p95_epe": float(np.percentile(epe, 95)),
            }
            all_means.append(rec["mean_epe"])
            all_p95.append(rec["p95_epe"])
        per_pair.append(rec)
    return {
        "name": name,
        "mean_of_mean_epe": float(np.mean(all_means)) if all_means else None,
        "mean_of_p95_epe": float(np.mean(all_p95)) if all_p95 else None,
        "per_pair": per_pair,
    }


def apply_depth_correction_full(depth: np.ndarray, g_maps: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if g_maps is None:
        return depth.astype(np.float32).copy()
    g_full = upsample_g_maps_np(g_maps, height, width)
    return (depth.astype(np.float32) * np.exp(g_full).astype(np.float32)).astype(np.float32)


def choose_depth_scale_fixed_point(depth: np.ndarray, percentile: float, precision: int, bit_depth: int) -> dict[str, Any]:
    max_code = (1 << bit_depth) - 1
    m = np.isfinite(depth) & (depth > 0)
    if not np.any(m):
        scale_real = 1.0 / max_code
    else:
        ref = float(np.percentile(depth[m], percentile))
        ref = max(ref, 1e-12)
        scale_real = ref / float(max_code)
    scale_int = max(1, int(round(scale_real * precision)))
    scale_real_q = scale_int / float(precision)
    return {
        "depth_scale": int(scale_int),
        "depth_scale_precision": int(precision),
        "depth_scale_real": float(scale_real_q),
        "depth_scale_percentile": float(percentile),
        "depth_bit_depth": int(bit_depth),
        "max_code": int(max_code),
    }


def write_depth_yuv420p10le_linear(path: Path, depth: np.ndarray, scale_meta: dict[str, Any]) -> dict[str, Any]:
    n, h, w = depth.shape
    if w % 2 or h % 2:
        raise ValueError("YUV420 output requires even width/height")
    ensure_parent(path)
    max_code = int(scale_meta["max_code"])
    scale = float(scale_meta["depth_scale_real"])
    neutral = np.uint16(512)
    clipped_total = 0
    with open(path, "wb") as f:
        for i in range(n):
            y = np.round(depth[i].astype(np.float64) / scale)
            clipped = (y < 0) | (y > max_code) | ~np.isfinite(y)
            clipped_total += int(np.count_nonzero(clipped))
            y = np.nan_to_num(y, nan=0.0, posinf=max_code, neginf=0.0)
            y = np.clip(y, 0, max_code).astype("<u2")
            uv = np.full((h // 2, w // 2), neutral, dtype="<u2")
            f.write(y.tobytes())
            f.write(uv.tobytes())
            f.write(uv.tobytes())
    return {
        **scale_meta,
        "depth_yuv": str(path),
        "depth_yuv_format": "yuv420p10le",
        "depth_yuv_semantics": "Y stores linear depth code = round(depth / depth_scale_real); U/V neutral 512",
        "clipped_samples_total": int(clipped_total),
        "total_samples": int(n * h * w),
    }


def write_camera_jsonl(
    path: Path,
    source_npz: Path,
    source_camera_jsonl: Path | None,
    frame_indices: np.ndarray,
    K_fixed: np.ndarray,
    r_abs: np.ndarray,
    t_abs: np.ndarray,
    depth_yuv_meta: dict[str, Any],
    cfg: OptimConfig,
    pair_list: list[PairSpec],
) -> None:
    ensure_parent(path)
    R_abs = np.stack([R_from_rvec_np(r_abs[i]) for i in range(len(r_abs))], axis=0)
    with open(path, "w", encoding="utf-8") as f:
        header = {
            "type": "header",
            "format": "fixedK_gop_abs_pose_nn_depth_v1",
            "source_npz": os.path.abspath(source_npz),
            "source_camera_jsonl": os.path.abspath(source_camera_jsonl) if source_camera_jsonl else None,
            "frame_count": int(len(frame_indices)),
            "frame_indices": frame_indices.astype(int).tolist(),
            "intrinsic_mode": "rap_fixed",
            "intrinsic": {
                "fx": float(K_fixed[0, 0]),
                "fy": float(K_fixed[1, 1]),
                "cx": float(K_fixed[0, 2]),
                "cy": float(K_fixed[1, 2]),
                "z_sign": float(cfg.z_sign),
            },
            "intrinsic_delta_order": [],
            "intrinsic_delta_bits_per_frame": 0,
            "pose_storage": {
                "absolute_pose": "camera_from_world in fixed-K canonical camera coordinates",
                "relative_pair_formula": "R_rel=R_ref@R_target.T; t_rel=t_ref-R_rel@t_target; X_ref=R_rel*X_target+t_rel",
                "adjacent_current_to_previous_fields": "also written for compatibility",
            },
            "depth_output": depth_yuv_meta,
            "optimization": {
                "summary": "Stage1 fixed-K R|t; Stage2 fixed-R t+tiny-NN multiplicative depth; Stage3 small-LR joint R/t/NN fine-tuning.",
                "config": asdict(cfg),
                "pair_count": len(pair_list),
                "pairs": [asdict(p) for p in pair_list],
            },
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i in range(len(frame_indices)):
            rec: dict[str, Any] = {
                "poc": int(i),
                "frame_idx": int(frame_indices[i]),
                "rvec_abs": as_float_list(r_abs[i]),
                "tvec_abs": as_float_list(t_abs[i]),
                "extrinsic_abs": np.concatenate([R_abs[i], t_abs[i].reshape(3, 1)], axis=1).astype(float).tolist(),
            }
            if i == 0:
                rec["rvec_current_to_previous"] = [0.0, 0.0, 0.0]
                rec["tvec_current_to_previous"] = [0.0, 0.0, 0.0]
            else:
                R_rel = R_abs[i - 1] @ R_abs[i].T
                t_rel = t_abs[i - 1] - R_rel @ t_abs[i]
                rec["rvec_current_to_previous"] = as_float_list(rvec_from_R(R_rel))
                rec["tvec_current_to_previous"] = as_float_list(t_rel)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# Main
# ============================================================

def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    npz_path = Path(args.npz)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    camera_jsonl_path = Path(args.camera_jsonl) if args.camera_jsonl else None
    out_prefix = Path(args.out_prefix)
    out_prefix = out_prefix.with_name(sanitize_windows_filename_component(out_prefix.name))

    out_npz = out_prefix.with_name(out_prefix.name + "_fixedK_gop_nn_geometry.npz")
    out_jsonl = out_prefix.with_name(out_prefix.name + "_fixedK_gop_nn_cam.jsonl")
    out_yuv = out_prefix.with_name(out_prefix.name + "_fixedK_gop_nn_depth_linear_yuv420p10le.yuv")
    out_manifest = out_prefix.with_name(out_prefix.name + "_fixedK_gop_nn_manifest.json")
    for p in [out_npz, out_jsonl, out_yuv, out_manifest]:
        if p.exists():
            if args.overwrite:
                p.unlink()
            else:
                raise RuntimeError(f"Output exists: {p}. Use --overwrite.")
        ensure_parent(p)

    log(f"Loading NPZ: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    for key in ["depth_original", "extrinsic", "intrinsic_original"]:
        if key not in data:
            raise KeyError(f"NPZ missing key: {key}")
    depth = data["depth_original"].astype(np.float32)
    E_abs = data["extrinsic"].astype(np.float64)
    K_orig = data["intrinsic_original"].astype(np.float64)
    frame_indices = data["frame_indices"].astype(np.int32) if "frame_indices" in data else np.arange(depth.shape[0], dtype=np.int32)
    n, h, w = depth.shape
    if args.width is not None and args.width != w:
        raise ValueError(f"--width {args.width} != NPZ width {w}")
    if args.height is not None and args.height != h:
        raise ValueError(f"--height {args.height} != NPZ height {h}")
    log(f"Loaded: depth={depth.shape}, extrinsic={E_abs.shape}, intrinsic={K_orig.shape}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    log(f"Using device={device}, dtype={dtype}")

    cfg = OptimConfig(
        device=str(device),
        dtype=args.dtype,
        seed=args.seed,
        fixed_center_mode=args.fixed_center_mode,
        z_sign=args.z_sign,
        include_adjacent=not args.no_adjacent,
        adjacent_bidirectional=args.adjacent_bidirectional,
        adjacent_weight=args.adjacent_weight,
        include_gop=not args.no_gop,
        bidirectional_gop=args.bidirectional_gop,
        gop_weight=args.gop_weight,
        extra_pairs=args.extra_pairs,
        sample_stride=args.sample_stride,
        max_samples_per_pair=args.max_samples_per_pair,
        stage1_iters=args.stage1_iters,
        stage1_lr_r=args.stage1_lr_r,
        stage1_lr_t=args.stage1_lr_t,
        stage1_rot_prior=args.stage1_rot_prior,
        stage1_t_prior=args.stage1_t_prior,
        stage2_iters=args.stage2_iters,
        stage2_lr_t=args.stage2_lr_t,
        stage2_lr_nn=args.stage2_lr_nn,
        stage2_t_prior=args.stage2_t_prior,
        stage3_iters=args.stage3_iters,
        stage3_lr_r=args.stage3_lr_r,
        stage3_lr_t=args.stage3_lr_t,
        stage3_lr_nn=args.stage3_lr_nn,
        stage3_rot_prior=args.stage3_rot_prior,
        stage3_t_prior=args.stage3_t_prior,
        stage3_depth_prior=args.stage3_depth_prior,
        nn_grid_cell=args.nn_grid_cell,
        nn_hidden_ch=args.nn_hidden_ch,
        nn_latent_ch=args.nn_latent_ch,
        nn_layers=args.nn_layers,
        depth_max_log_scale=args.depth_max_log_scale,
        depth_l2=args.depth_l2,
        depth_spatial_smooth=args.depth_spatial_smooth,
        depth_temporal_smooth=args.depth_temporal_smooth,
        depth_edge_alpha=args.depth_edge_alpha,
        f_scale=args.f_scale,
        print_every=args.print_every,
        depth_scale_precision=args.depth_scale_precision,
        depth_scale_percentile=args.depth_scale_percentile,
    )

    log("Building fixed RAP-level K")
    K_fixed = make_fixed_intrinsic(K_orig, w, h, cfg.fixed_center_mode)
    log(f"K_fixed: fx={K_fixed[0,0]:.6f}, fy={K_fixed[1,1]:.6f}, cx={K_fixed[0,2]:.6f}, cy={K_fixed[1,2]:.6f}")

    log("Initializing absolute fixed-K rigid poses from VGGT affine cameras")
    r_init, t_init = init_fixedK_absolute_poses(E_abs, K_orig, K_fixed)

    log("Building codec-relevant pair list")
    pair_list = build_pair_list(
        n=n,
        include_adjacent=cfg.include_adjacent,
        adjacent_bidirectional=cfg.adjacent_bidirectional,
        adjacent_weight=cfg.adjacent_weight,
        include_gop=cfg.include_gop,
        bidirectional_gop=cfg.bidirectional_gop,
        gop_weight=cfg.gop_weight,
        extra_pairs=cfg.extra_pairs,
    )
    if not pair_list:
        raise RuntimeError("No training pairs. Enable GOP/adjacent or pass --extra-pairs.")
    log(f"Pair count: {len(pair_list)}")
    for p in pair_list[:60]:
        log(f"  pair {p.target}->{p.ref}, weight={p.weight:.3g}, kind={p.kind}")
    if len(pair_list) > 60:
        log(f"  ... {len(pair_list)-60} more pairs")

    log("Creating fixed-K ray grid and exact VGGT target samples")
    rays_fixed = make_rays_np(K_fixed, w, h, z_sign=cfg.z_sign)
    pair_cache_np = create_pair_cache(
        pair_list,
        depth,
        E_abs,
        K_orig,
        K_fixed,
        rays_fixed,
        sample_stride=cfg.sample_stride,
        max_samples_per_pair=cfg.max_samples_per_pair,
        z_sign=cfg.z_sign,
        seed=cfg.seed,
    )
    if not pair_cache_np:
        raise RuntimeError("All pairs had zero valid samples.")
    pair_cache_t = cache_to_torch(pair_cache_np, device=device, dtype=dtype)

    log("Building low-res NN depth features")
    feat_np, wx_np, wy_np = build_depth_nn_features_np(depth, grid_cell=cfg.nn_grid_cell, include_xy=True)
    log(f"NN feature grid: features={feat_np.shape}, wx={wx_np.shape}, wy={wy_np.shape}")
    features = torch.from_numpy(feat_np).to(device=device, dtype=dtype)
    wx_t = torch.from_numpy(wx_np).to(device=device, dtype=dtype)
    wy_t = torch.from_numpy(wy_np).to(device=device, dtype=dtype)

    log("Evaluating initial fixed-K affine-to-rigid approximation")
    init_eval = evaluate_on_cache("init", pair_cache_np, r_init, t_init, K_fixed, w, h, cfg.z_sign, None)
    log(f"Initial EPE: mean={init_eval['mean_of_mean_epe']:.6f}, p95={init_eval['mean_of_p95_epe']:.6f}")

    r_stage1, t_stage1, stage1_hist = optimize_stage1_rt(pair_cache_t, r_init, t_init, K_fixed, w, h, cfg, device, dtype)
    stage1_eval = evaluate_on_cache("stage1_rt", pair_cache_np, r_stage1, t_stage1, K_fixed, w, h, cfg.z_sign, None)
    log(f"Stage1 EPE: mean={stage1_eval['mean_of_mean_epe']:.6f}, p95={stage1_eval['mean_of_p95_epe']:.6f}")

    if cfg.stage2_iters > 0:
        t_stage2, net, latent_stage2, g_stage2, stage2_hist = optimize_stage2_t_nn(
            pair_cache_t, r_stage1, t_stage1, K_fixed, features, wx_t, wy_t, cfg, w, h, device, dtype
        )
    else:
        log("Stage 2 skipped because --stage2-iters 0")
        t_stage2 = t_stage1.copy()
        net = DepthCorrectionNet(feature_ch=features.shape[1], latent_ch=cfg.nn_latent_ch, hidden_ch=cfg.nn_hidden_ch, layers=cfg.nn_layers).to(device=device, dtype=dtype)
        latent_stage2 = torch.zeros((n, cfg.nn_latent_ch, features.shape[2], features.shape[3]), device=device, dtype=dtype)
        g_stage2 = np.zeros((n, 1, features.shape[2], features.shape[3]), dtype=np.float32)
        stage2_hist = []

    stage2_eval = evaluate_on_cache("stage2_t_nn_depth", pair_cache_np, r_stage1, t_stage2, K_fixed, w, h, cfg.z_sign, g_stage2)
    log(f"Stage2 EPE: mean={stage2_eval['mean_of_mean_epe']:.6f}, p95={stage2_eval['mean_of_p95_epe']:.6f}")

    if cfg.stage3_iters > 0:
        r_stage3, t_stage3, net, latent_stage3, g_stage3, stage3_hist = optimize_stage3_joint(
            pair_cache_t, r_stage1, t_stage2, net, latent_stage2, g_stage2, K_fixed, features, wx_t, wy_t, cfg, w, h, device, dtype
        )
    else:
        log("Stage 3 skipped because --stage3-iters 0")
        r_stage3 = r_stage1.copy()
        t_stage3 = t_stage2.copy()
        latent_stage3 = latent_stage2
        g_stage3 = g_stage2
        stage3_hist = []

    stage3_eval = evaluate_on_cache("stage3_joint", pair_cache_np, r_stage3, t_stage3, K_fixed, w, h, cfg.z_sign, g_stage3)
    log(f"Stage3 EPE: mean={stage3_eval['mean_of_mean_epe']:.6f}, p95={stage3_eval['mean_of_p95_epe']:.6f}")

    log("Applying final NN depth correction and writing depth YUV")
    depth_canonical = apply_depth_correction_full(depth, g_stage3, h, w)
    depth_scale_meta = choose_depth_scale_fixed_point(depth_canonical, cfg.depth_scale_percentile, cfg.depth_scale_precision, 10)
    depth_yuv_meta = write_depth_yuv420p10le_linear(out_yuv, depth_canonical, depth_scale_meta)

    log("Writing camera JSONL")
    write_camera_jsonl(out_jsonl, npz_path, camera_jsonl_path, frame_indices, K_fixed, r_stage3, t_stage3, depth_yuv_meta, cfg, pair_list)

    log("Saving NPZ")
    payload: dict[str, Any] = {
        "frame_indices": frame_indices.astype(np.int32),
        "K_fixed": K_fixed.astype(np.float32),
        "rvec_abs_init": r_init.astype(np.float32),
        "tvec_abs_init": t_init.astype(np.float32),
        "rvec_abs_stage1_rt": r_stage1.astype(np.float32),
        "tvec_abs_stage1_rt": t_stage1.astype(np.float32),
        "rvec_abs_stage2_t_nn": r_stage1.astype(np.float32),
        "tvec_abs_stage2_t_nn": t_stage2.astype(np.float32),
        "rvec_abs_final": r_stage3.astype(np.float32),
        "tvec_abs_final": t_stage3.astype(np.float32),
        "depth_canonical": depth_canonical.astype(np.float32),
        "depth_log_scale_grid_stage2": g_stage2.astype(np.float32),
        "depth_log_scale_grid_final": g_stage3.astype(np.float32),
        "depth_log_scale_grid_shape": np.asarray(g_stage3.shape, dtype=np.int32),
        "config_json": np.asarray(json.dumps(asdict(cfg), ensure_ascii=False), dtype=object),
        "pairs_json": np.asarray(json.dumps([asdict(p) for p in pair_list], ensure_ascii=False), dtype=object),
        "init_eval_json": np.asarray(json.dumps(init_eval, ensure_ascii=False), dtype=object),
        "stage1_eval_json": np.asarray(json.dumps(stage1_eval, ensure_ascii=False), dtype=object),
        "stage2_eval_json": np.asarray(json.dumps(stage2_eval, ensure_ascii=False), dtype=object),
        "stage3_eval_json": np.asarray(json.dumps(stage3_eval, ensure_ascii=False), dtype=object),
        "stage1_history_json": np.asarray(json.dumps(stage1_hist, ensure_ascii=False), dtype=object),
        "stage2_history_json": np.asarray(json.dumps(stage2_hist, ensure_ascii=False), dtype=object),
        "stage3_history_json": np.asarray(json.dumps(stage3_hist, ensure_ascii=False), dtype=object),
        "nn_feature_grid": feat_np.astype(np.float32),
        "nn_edge_weight_x": wx_np.astype(np.float32),
        "nn_edge_weight_y": wy_np.astype(np.float32),
    }
    if args.save_nn_state:
        # torch state_dict cannot be stored directly in a portable npz; save separate .pt.
        out_pt = out_prefix.with_name(out_prefix.name + "_fixedK_gop_nn_state.pt")
        torch.save({
            "net_state_dict": net.state_dict(),
            "latent": latent_stage3.detach().cpu(),
            "K_fixed": torch.from_numpy(K_fixed.astype(np.float32)),
            "config": asdict(cfg),
        }, out_pt)
        payload["nn_state_pt"] = np.asarray(str(out_pt), dtype=object)
    if args.save_original_debug:
        payload["depth_original"] = depth.astype(np.float32)
        payload["intrinsic_original"] = K_orig.astype(np.float32)
        payload["extrinsic_original"] = E_abs.astype(np.float32)
    if args.compressed_npz:
        np.savez_compressed(out_npz, **payload)
    else:
        np.savez(out_npz, **payload)

    manifest = {
        "source_npz": os.path.abspath(npz_path),
        "source_camera_jsonl": os.path.abspath(camera_jsonl_path) if camera_jsonl_path else None,
        "outputs": {
            "geometry_npz": os.path.abspath(out_npz),
            "camera_jsonl": os.path.abspath(out_jsonl),
            "depth_yuv": os.path.abspath(out_yuv),
            "manifest": os.path.abspath(out_manifest),
        },
        "frame_count": int(n),
        "size": {"width": int(w), "height": int(h)},
        "K_fixed": K_fixed.astype(float).tolist(),
        "config": asdict(cfg),
        "pair_count": len(pair_list),
        "pairs": [asdict(p) for p in pair_list],
        "eval": {
            "init": init_eval,
            "stage1_rt": stage1_eval,
            "stage2_t_nn_depth": stage2_eval,
            "stage3_joint": stage3_eval,
        },
        "depth_yuv": depth_yuv_meta,
        "notes": [
            "Pair direction is target->reference; backward projection uses target-frame depth.",
            "GOP/hierarchical anchor pairs are included directly in the optimization loss.",
            "Depth correction is produced by a tiny low-res NN during encoder-side optimization only.",
            "Decoder does not need the NN; it only receives the final corrected depth YUV and fixed-K camera JSONL.",
            "Intrinsic is fixed once per RAP; no per-frame intrinsic delta is written.",
        ],
    }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("============================================================")
    print("Fixed-K GOP NN-depth optimization done")
    print("============================================================")
    print(f"input npz       : {npz_path}")
    print(f"frames          : {n}")
    print(f"size            : {w}x{h}")
    print(f"K fixed         : fx={K_fixed[0,0]:.6f}, fy={K_fixed[1,1]:.6f}, cx={K_fixed[0,2]:.6f}, cy={K_fixed[1,2]:.6f}")
    print(f"pairs           : {len(pair_list)}")
    print(f"NN grid         : {g_stage3.shape[2]}x{g_stage3.shape[3]}  cell~{cfg.nn_grid_cell}")
    print("------------------------------------------------------------")
    print(f"init    mean/p95: {init_eval['mean_of_mean_epe']:.6f} / {init_eval['mean_of_p95_epe']:.6f} px")
    print(f"stage1  mean/p95: {stage1_eval['mean_of_mean_epe']:.6f} / {stage1_eval['mean_of_p95_epe']:.6f} px")
    print(f"stage2  mean/p95: {stage2_eval['mean_of_mean_epe']:.6f} / {stage2_eval['mean_of_p95_epe']:.6f} px")
    print(f"stage3  mean/p95: {stage3_eval['mean_of_mean_epe']:.6f} / {stage3_eval['mean_of_p95_epe']:.6f} px")
    print("------------------------------------------------------------")
    print(f"geometry npz    : {out_npz}")
    print(f"camera jsonl    : {out_jsonl}")
    print(f"depth yuv       : {out_yuv}")
    print(f"manifest        : {out_manifest}")
    print("============================================================")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-K R|t + constrained tiny-NN depth correction optimized over codec GOP pairs")
    p.add_argument("--npz", required=True, help="VGGT-Omega output NPZ")
    p.add_argument("--camera-jsonl", default=None, help="Optional source camera JSONL")
    p.add_argument("--out-prefix", required=True)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--device", default="auto", help="auto/cuda/cpu")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--fixed-center-mode", choices=["image-center", "median", "first"], default="image-center")
    p.add_argument("--z-sign", type=float, default=1.0)

    # Pair construction.
    p.add_argument("--no-adjacent", action="store_true", help="Disable adjacent i->i-1 pairs")
    p.add_argument("--adjacent-bidirectional", action="store_true", help="Also include i-1->i adjacent reverse pairs")
    p.add_argument("--adjacent-weight", type=float, default=0.5)
    p.add_argument("--no-gop", action="store_true", help="Disable recursively generated GOP/hierarchical pairs")
    p.add_argument("--bidirectional-gop", action="store_true", default=True, help="Include both current->anchor and anchor->current hierarchical pairs")
    p.add_argument("--no-bidirectional-gop", dest="bidirectional_gop", action="store_false")
    p.add_argument("--gop-weight", type=float, default=1.0)
    p.add_argument("--extra-pairs", default=None, help="Additional target:ref[:weight] pairs. Example: '0:16:2,32:16:2,16:8'")

    # Sampling.
    p.add_argument("--sample-stride", type=int, default=8)
    p.add_argument("--max-samples-per-pair", type=int, default=60000, help="0 means use all stride samples")

    # Stage 1.
    p.add_argument("--stage1-iters", type=int, default=300)
    p.add_argument("--stage1-lr-r", type=float, default=1e-3)
    p.add_argument("--stage1-lr-t", type=float, default=1e-3)
    p.add_argument("--stage1-rot-prior", type=float, default=1e-6)
    p.add_argument("--stage1-t-prior", type=float, default=1e-6)

    # Stage 2.
    p.add_argument("--stage2-iters", type=int, default=300)
    p.add_argument("--stage2-lr-t", type=float, default=5e-4)
    p.add_argument("--stage2-lr-nn", type=float, default=2e-3)
    p.add_argument("--stage2-t-prior", type=float, default=1e-4)

    # Stage 3: conservative joint fine-tuning from Stage 2.
    p.add_argument("--stage3-iters", type=int, default=150)
    p.add_argument("--stage3-lr-r", type=float, default=3e-5)
    p.add_argument("--stage3-lr-t", type=float, default=3e-5)
    p.add_argument("--stage3-lr-nn", type=float, default=3e-4)
    p.add_argument("--stage3-rot-prior", type=float, default=1e-3)
    p.add_argument("--stage3-t-prior", type=float, default=1e-3)
    p.add_argument("--stage3-depth-prior", type=float, default=0.5, help="Regularize Stage3 g map toward Stage2 g map")

    # NN depth correction.
    p.add_argument("--nn-grid-cell", type=int, default=64, help="Low-res correction grid cell size in source pixels. 64 -> ~17x30 at 1080p")
    p.add_argument("--nn-hidden-ch", type=int, default=24)
    p.add_argument("--nn-latent-ch", type=int, default=2)
    p.add_argument("--nn-layers", type=int, default=3)
    p.add_argument("--depth-max-log-scale", type=float, default=0.15, help="g=max*tanh(raw); exp(g) is multiplicative depth scale")
    p.add_argument("--depth-l2", type=float, default=0.1)
    p.add_argument("--depth-spatial-smooth", type=float, default=0.5)
    p.add_argument("--depth-temporal-smooth", type=float, default=0.05)
    p.add_argument("--depth-edge-alpha", type=float, default=1.0, help="Higher value weakens smoothing more strongly at VGGT depth edges")

    # Loss / output.
    p.add_argument("--f-scale", type=float, default=1.0, help="Charbonnier soft scale in pixels")
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--depth-scale-precision", type=int, default=100000)
    p.add_argument("--depth-scale-percentile", type=float, default=99.9)
    p.add_argument("--compressed-npz", action="store_true")
    p.add_argument("--save-original-debug", action="store_true")
    p.add_argument("--save-nn-state", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if args.nn_grid_cell <= 0:
        raise ValueError("--nn-grid-cell must be positive")
    run(args)


if __name__ == "__main__":
    main()
