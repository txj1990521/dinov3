#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv3 (Meta official) + Hybrid FAISS builder
- Global index: multi-view aggregation (max-pool + power-norm) -> 1 embedding / image
- Patch indexes:
  - Stripe patches for extreme aspect-ratio images
  - Grid patches for normal images
- FAISS:
  - FlatIP for small sets
  - IVF-Flat (global) / IVF-PQ (patch) for large sets

Outputs (in OUT_DIR):
- global.index
- global_img_paths.npy             (img_id -> path)

- patch_stripe.index
- patch_stripe_meta.npy            (patch_id -> [img_id,x1,y1,x2,y2,win,ptype,pos])

- patch_grid.index
- patch_grid_meta.npy              (patch_id -> [img_id,x1,y1,x2,y2,win,ptype,pos])

Notes:
- This script assumes you installed Meta's dinov3 repo in editable mode so imports work.
  e.g.:
    git clone https://github.com/facebookresearch/dinov3
    pip install -e ./dinov3
"""

import os
import math
import random
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm

# =========================
# CONFIG: 只改这里
# =========================
CKPT = r"pre_model/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DATA_ROOT = r"D:\zhanlan\new_data"
OUT_DIR = r"D:\zhanlan\faiss_database_dinov3_hybrid"

GLOBAL_INDEX = os.path.join(OUT_DIR, "global.index")
GLOBAL_META  = os.path.join(OUT_DIR, "global_img_paths.npy")

PATCH_STRIPE_INDEX = os.path.join(OUT_DIR, "patch_stripe.index")
PATCH_GRID_INDEX   = os.path.join(OUT_DIR, "patch_grid.index")
PATCH_STRIPE_META  = os.path.join(OUT_DIR, "patch_stripe_meta.npy")
PATCH_GRID_META    = os.path.join(OUT_DIR, "patch_grid_meta.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# DINOv3 preprocess (Meta style)
# =========================
# DINO family commonly uses (x/255 - 0.5)/0.5
DINO_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
DINO_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)

# =========================
# Global multi-view config
# =========================
VIEWS_PER_IMAGE = 12
RESIZE_SHORT = 256
CROP_SIZE = 224
VIEW_PLAN = [
    (0,   1, 5),
    (-15, 1, 1),
    (15,  1, 1),
    (-30, 1, 0),
    (30,  1, 0),
]
VIEW_BATCH = 256  # batch in "views"

# =========================
# Patch tiling config
# =========================
PATCH_ENC_SIZE = 224

# stripe decision + params
STRIPE_AR_THR = 2.5          # >= this => treat as stripe image
STRIPE_LONG_EDGE = 1024      # stable resize reference for seeding
STRIPE_MAX_PATCHES = 24
STRIPE_WIN_H = 224           # length along long edge
STRIPE_STRIDE = 48
STRIPE_CENTER_FRAC = 0.92
STRIPE_JITTER = 8

# grid params
PATCH_SIZES = (224, 320, 384, 448)
STRIDE_RATIO = 0.5
MAX_LONG = 1024
MAX_PATCHES_PER_IMAGE = 64

# =========================
# IVF/PQ settings
# =========================
IVF_SEED = 123
GLOBAL_FLAT_THRESHOLD = 20000        # images
PATCH_FLAT_THRESHOLD_VECS = 200000   # patches
PQ_M = 32
PQ_NBITS = 8

# =========================
# utils
# =========================
def ensure_dir_for_file(fp: str):
    Path(fp).parent.mkdir(parents=True, exist_ok=True)

def imread_unicode(p: str):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def list_images_recursive(root: str):
    root = Path(root)
    paths = [p for p in root.rglob("*") if p.is_file() and is_image(p)]
    paths.sort()
    return paths

def resize_long_edge(img_bgr: np.ndarray, max_long=1024):
    h, w = img_bgr.shape[:2]
    long_ = max(h, w)
    if long_ <= max_long:
        return img_bgr
    scale = max_long / long_
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

def seed_from_image(img_bgr: np.ndarray, base: int = 0) -> int:
    # fast deterministic seed from image bytes (after downscale)
    small = cv2.resize(img_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    h = hashlib.md5(small.tobytes()).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff

def power_norm_torch(x: torch.Tensor, eps: float = 1e-12):
    return torch.sign(x) * torch.sqrt(torch.clamp(torch.abs(x), min=eps))

# =========================
# preprocess for global views
# =========================
def resize_short_edge(img_rgb: np.ndarray, short=256):
    h, w = img_rgb.shape[:2]
    if min(h, w) == short:
        return img_rgb
    scale = short / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

def center_crop(img_rgb: np.ndarray, size=224):
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return img_rgb[y1:y1+size, x1:x1+size]

def random_crop(img_rgb: np.ndarray, size=224, rng=None):
    h, w = img_rgb.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        h, w = img_rgb.shape[:2]
    if rng is None:
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
    else:
        y = int(rng.integers(0, h - size + 1))
        x = int(rng.integers(0, w - size + 1))
    return img_rgb[y:y+size, x:x+size]

def rotate_bound(img_rgb: np.ndarray, deg: float):
    if deg == 0:
        return img_rgb
    h, w = img_rgb.shape[:2]
    cX, cY = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cX, cY), deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img_rgb, M, (nW, nH),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)

def to_tensor_from_rgb_dino(img_rgb_crop: np.ndarray):
    x = img_rgb_crop.astype(np.float32) / 255.0
    x = (x - DINO_MEAN) / DINO_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

# =========================
# Global views -> embedding
# =========================
@torch.no_grad()
def make_views_for_global(img_bgr: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = resize_short_edge(img_rgb, RESIZE_SHORT)

    views = []
    for deg, n_center, n_rand in VIEW_PLAN:
        rot = rotate_bound(img_rgb, deg)
        for _ in range(n_center):
            views.append(to_tensor_from_rgb_dino(center_crop(rot, CROP_SIZE)))
        for _ in range(n_rand):
            views.append(to_tensor_from_rgb_dino(random_crop(rot, CROP_SIZE, rng=rng)))
    return views[:VIEWS_PER_IMAGE]

@torch.no_grad()
def aggregate_views_to_one(feats_view: torch.Tensor):
    agg = feats_view.max(dim=0).values
    agg = power_norm_torch(agg)
    agg = F.normalize(agg.unsqueeze(0), p=2, dim=1).squeeze(0)
    return agg

# =========================
# Patch tiling (stripe + grid)
# =========================
def sample_windows_deterministic(windows, k: int, seed: int):
    if len(windows) <= k:
        return windows
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(windows), size=k, replace=False)
    idx = np.sort(idx)
    return [windows[i] for i in idx.tolist()]

def gen_stripe_windows(
    H: int, W: int,
    max_patches: int,
    seed: int,
    win_w: int,
    win_h: int,
    stride: int,
    center_frac: float = 0.92,
    jitter: int = 8,
):
    rng = np.random.default_rng(seed)
    vertical = (H >= W)
    windows = []

    if vertical:
        Wu = int(round(W * center_frac))
        x0 = max(0, (W - Wu) // 2)
        ww = min(win_w, Wu)
        hh = min(win_h, H)
        if ww < 16 or hh < 16:
            return []
        x_base = x0 + max(0, (Wu - ww) // 2)

        y = 0
        while y + hh <= H and len(windows) < max_patches:
            dx = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            x1 = int(np.clip(x_base + dx, x0, x0 + Wu - ww))
            y1 = int(y)
            x2 = x1 + ww
            y2 = y1 + hh
            center = (y1 + y2) * 0.5
            pos = int(np.clip((center / max(1.0, H)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))
            y += stride

        if windows and windows[-1][3] < H:
            y1 = max(0, H - hh)
            x1 = windows[-1][0]
            x2 = x1 + ww
            y2 = y1 + hh
            center = (y1 + y2) * 0.5
            pos = int(np.clip((center / max(1.0, H)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))

    else:
        Hu = int(round(H * center_frac))
        y0 = max(0, (H - Hu) // 2)
        hh = min(win_w, Hu)  # thickness
        ww = min(win_h, W)   # length
        if hh < 16 or ww < 16:
            return []
        y_base = y0 + max(0, (Hu - hh) // 2)

        x = 0
        while x + ww <= W and len(windows) < max_patches:
            dy = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            y1 = int(np.clip(y_base + dy, y0, y0 + Hu - hh))
            x1 = int(x)
            x2 = x1 + ww
            y2 = y1 + hh
            center = (x1 + x2) * 0.5
            pos = int(np.clip((center / max(1.0, W)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))
            x += stride

        if windows and windows[-1][2] < W:
            x1 = max(0, W - ww)
            y1 = windows[-1][1]
            x2 = x1 + ww
            y2 = y1 + hh
            center = (x1 + x2) * 0.5
            pos = int(np.clip((center / max(1.0, W)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))

    return windows

def gen_patch_windows_unified(
    img_bgr: np.ndarray,
    max_long: int,
    stripe_ar_thr: float,
    seed_base: int,
    stripe_max_patches: int,
    stripe_win_h: int,
    stripe_stride: int,
    stripe_center_frac: float,
    stripe_jitter: int,
    grid_sizes: Tuple[int, ...],
    grid_stride_ratio: float,
    max_patches: int,
):
    img = resize_long_edge(img_bgr, max_long=max_long)
    H, W = img.shape[:2]
    ar = max(W / (H + 1e-6), H / (W + 1e-6))
    is_stripe = (ar >= stripe_ar_thr)

    seed = seed_from_image(img, base=seed_base)

    windows = []
    if is_stripe:
        # width: clamp based on short side
        win_w = int(np.clip(0.85 * min(H, W), 160, 256))
        windows = gen_stripe_windows(
            H, W,
            max_patches=stripe_max_patches,
            seed=seed,
            win_w=win_w,
            win_h=min(stripe_win_h, max(H, W)),
            stride=max(32, stripe_stride),
            center_frac=stripe_center_frac,
            jitter=stripe_jitter,
        )
        if len(windows) < 6:
            ww = min(max(win_w, 160), W)
            hh = min(max(stripe_win_h, 256), H)
            x1 = max(0, (W - ww) // 2)
            y1 = max(0, (H - hh) // 2)
            windows.append((x1, y1, x1 + ww, y1 + hh, int(max(ww, hh)), 1, 5000))
    else:
        for win in grid_sizes:
            if H < win or W < win:
                continue
            stride = max(1, int(round(win * grid_stride_ratio)))
            for y1 in range(0, H - win + 1, stride):
                for x1 in range(0, W - win + 1, stride):
                    windows.append((x1, y1, x1 + win, y1 + win, win, 0, 5000))

        for win in (512, 384, 256, 224):
            if H >= win and W >= win:
                cx1 = (W - win) // 2
                cy1 = (H - win) // 2
                windows.append((cx1, cy1, cx1 + win, cy1 + win, win, 0, 5000))
                break

        windows = list(dict.fromkeys(windows))
        if max_patches and len(windows) > max_patches:
            windows = sample_windows_deterministic(windows, max_patches, seed=seed)

    return img, windows, is_stripe

def patch_to_model_input(patch_bgr: np.ndarray, enc_size: int):
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    patch_rgb = cv2.resize(patch_rgb, (enc_size, enc_size), interpolation=cv2.INTER_LINEAR)
    return to_tensor_from_rgb_dino(patch_rgb)

# =========================
# DINOv3 model
# =========================
@torch.no_grad()
def  build_dinov3_vitl16(ckpt_path: str, device: str):
    """
    Meta official dinov3 vit-large patch16.
    Import path may vary by repo version; this is the common one.
    """
    try:
        from dinov3.models.vision_transformer import vit_large
    except Exception as e:
        raise RuntimeError(
            "Cannot import dinov3. Please install Meta's dinov3 repo (pip install -e)."
        ) from e

    model = vit_large(patch_size=16)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # handle common formats
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    # clean prefixes
    cleaned = {}
    for k, v in sd.items():
        kk = k
        for pref in ("module.", "model.", "backbone."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[MODEL] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(device)
    return model

@torch.no_grad()
@torch.no_grad()
def extract_dino_embedding(model, batch_tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns (B, D) normalized embedding. Prefer CLS token.
    Compatible with different dinov3 repo versions / output formats.
    """
    feats = model.forward_features(batch_tensor)

    if feats is None:
        raise RuntimeError(
            f"model.forward_features returned None. "
            f"batch shape={tuple(batch_tensor.shape)} dtype={batch_tensor.dtype} device={batch_tensor.device}"
        )

    # Some versions may return object with attributes, not dict
    # dict case
    if isinstance(feats, dict):
        # common keys: 'x', 'cls', 'tokens', 'x_norm_clstoken', etc.
        for k in ("x_norm_clstoken", "cls", "clstoken", "x", "tokens"):
            if k in feats and feats[k] is not None:
                feats = feats[k]
                break
        else:
            # pick the last non-None value
            vals = [v for v in feats.values() if v is not None]
            if not vals:
                raise RuntimeError(f"forward_features dict has no usable values. keys={list(feats.keys())}")
            feats = vals[-1]

    # tuple/list case
    if isinstance(feats, (tuple, list)):
        # pick last non-None
        vals = [v for v in feats if v is not None]
        if not vals:
            raise RuntimeError("forward_features returned tuple/list but all elements are None")
        feats = vals[-1]

    if not torch.is_tensor(feats):
        raise RuntimeError(f"forward_features returned unsupported type: {type(feats)}")

    # Now feats is a Tensor
    if feats.dim() == 3:
        out = feats[:, 0, :]  # CLS
    elif feats.dim() == 2:
        out = feats
    elif feats.dim() == 4:
        out = feats.mean(dim=(2, 3))
    else:
        raise RuntimeError(f"Unsupported feature tensor shape: {tuple(feats.shape)}")

    out = F.normalize(out, p=2, dim=1)
    return out

# =========================
# FAISS helpers
# =========================
def pick_nlist(N: int) -> int:
    nlist = int(4 * np.sqrt(max(N, 1)))
    nlist = max(1024, nlist)
    nlist = min(65536, nlist)
    nlist = min(nlist, max(1, N // 20))
    return max(1, nlist)

def pick_train_size(N: int, nlist: int) -> int:
    t = min(N, max(nlist, 100 * nlist))
    t = min(t, 500_000)
    return int(t)

def build_ivfflat_ip(feats: np.ndarray, seed: int):
    feats = feats.astype("float32")
    N, D = feats.shape
    nlist = pick_nlist(N)
    train_size = pick_train_size(N, nlist)
    m = min(train_size, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=m, replace=False)
    train_x = feats[idx]

    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)
    print(f"[IVF-Flat] training on {m}, nlist={nlist}")
    index.train(train_x)
    index.add(feats)
    index.nprobe = min(64, index.nlist)
    return index

def build_ivfpq_ip(feats: np.ndarray, seed: int, m: int = PQ_M, nbits: int = PQ_NBITS):
    feats = feats.astype("float32")
    N, D = feats.shape
    if D % m != 0:
        raise ValueError(f"PQ_M={m} must divide D={D}. Change PQ_M.")
    nlist = pick_nlist(N)
    train_size = pick_train_size(N, nlist)
    tr = min(train_size, N)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=tr, replace=False)
    train_x = feats[idx]

    quantizer = faiss.IndexFlatIP(D)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
    print(f"[IVF-PQ] training on {tr}, nlist={nlist}, m={m}, nbits={nbits}")
    index.train(train_x)
    index.add(feats)
    index.nprobe = min(64, index.nlist)
    return index

# =========================
# main
# =========================
def main():
    print("[INFO] device:", DEVICE)
    ensure_dir_for_file(GLOBAL_INDEX)
    ensure_dir_for_file(PATCH_STRIPE_INDEX)
    ensure_dir_for_file(PATCH_GRID_INDEX)

    model = build_dinov3_vitl16(CKPT, DEVICE)

    paths = list_images_recursive(DATA_ROOT)
    print(f"[INFO] Found {len(paths)} images under {DATA_ROOT}")

    # Build a canonical valid image list
    valid_images = []
    for p in paths:
        img = imread_unicode(str(p))
        if img is None:
            continue
        valid_images.append(str(p))
    print(f"[INFO] Valid images = {len(valid_images)}")

    # ---------- GLOBAL feats (streaming) ----------
    global_feats_chunks = []
    img_paths_kept = []

    IMG_BATCH = max(1, VIEW_BATCH // max(1, VIEWS_PER_IMAGE))
    print(f"[GLOBAL] VIEWS_PER_IMAGE={VIEWS_PER_IMAGE}, VIEW_BATCH={VIEW_BATCH}, IMG_BATCH={IMG_BATCH}")

    img_batch_paths = []
    img_batch_views = []

    @torch.no_grad()
    def flush_global_batch():
        nonlocal img_batch_paths, img_batch_views
        if not img_batch_paths:
            return
        flat_views = []
        offsets = [0]
        for vs in img_batch_views:
            flat_views.extend(vs)
            offsets.append(len(flat_views))

        bt = torch.stack(flat_views, dim=0).to(DEVICE)
        feats_v = extract_dino_embedding(model, bt).cpu()  # (sumV, D)

        feats_img = []
        for k in range(len(img_batch_paths)):
            s, e = offsets[k], offsets[k+1]
            fv = feats_v[s:e]
            agg = aggregate_views_to_one(fv)
            feats_img.append(agg.unsqueeze(0))

        feats_img = torch.cat(feats_img, dim=0).numpy().astype("float32")
        global_feats_chunks.append(feats_img)
        img_paths_kept.extend(img_batch_paths)

        img_batch_paths, img_batch_views = [], []

    # ---------- PATCH feats (streaming) ----------
    PATCH_BATCH = 256

    patch_buf_tensors_s, patch_buf_meta_s = [], []
    patch_feats_chunks_s, patch_meta_list_s = [], []

    patch_buf_tensors_g, patch_buf_meta_g = [], []
    patch_feats_chunks_g, patch_meta_list_g = [], []

    @torch.no_grad()
    def flush_patch_batch_stripe():
        nonlocal patch_buf_tensors_s, patch_buf_meta_s
        if not patch_buf_tensors_s:
            return
        bt = torch.stack(patch_buf_tensors_s).to(DEVICE)
        feats = extract_dino_embedding(model, bt).cpu().numpy().astype("float32")
        patch_feats_chunks_s.append(feats)
        patch_meta_list_s.extend(patch_buf_meta_s)
        patch_buf_tensors_s.clear()
        patch_buf_meta_s.clear()

    @torch.no_grad()
    def flush_patch_batch_grid():
        nonlocal patch_buf_tensors_g, patch_buf_meta_g
        if not patch_buf_tensors_g:
            return
        bt = torch.stack(patch_buf_tensors_g).to(DEVICE)
        feats = extract_dino_embedding(model, bt).cpu().numpy().astype("float32")
        patch_feats_chunks_g.append(feats)
        patch_meta_list_g.extend(patch_buf_meta_g)
        patch_buf_tensors_g.clear()
        patch_buf_meta_g.clear()

    for idx, path in enumerate(tqdm(valid_images, desc="Building index"), 1):
        img = imread_unicode(path)
        if img is None:
            continue

        img_id = idx - 1

        # ---- GLOBAL ----
        seed_src = resize_long_edge(img, max_long=STRIPE_LONG_EDGE)
        seed_g = seed_from_image(seed_src, base=0)
        random.seed(seed_g)

        views = make_views_for_global(seed_src, seed=seed_g)
        if views:
            img_batch_paths.append(path)
            img_batch_views.append(views)
            if len(img_batch_paths) >= IMG_BATCH:
                flush_global_batch()

        # ---- PATCH ----
        img_resized, windows, is_stripe = gen_patch_windows_unified(
            img,
            max_long=MAX_LONG,
            stripe_ar_thr=STRIPE_AR_THR,
            seed_base=999,
            stripe_max_patches=STRIPE_MAX_PATCHES,
            stripe_win_h=STRIPE_WIN_H,
            stripe_stride=STRIPE_STRIDE,
            stripe_center_frac=STRIPE_CENTER_FRAC,
            stripe_jitter=STRIPE_JITTER,
            grid_sizes=PATCH_SIZES,
            grid_stride_ratio=STRIDE_RATIO,
            max_patches=MAX_PATCHES_PER_IMAGE,
        )

        for (x1, y1, x2, y2, win, ptype, pos) in windows:
            patch = img_resized[y1:y2, x1:x2]
            t = patch_to_model_input(patch, enc_size=PATCH_ENC_SIZE)

            if ptype == 1:
                patch_buf_tensors_s.append(t)
                patch_buf_meta_s.append((img_id, x1, y1, x2, y2, win, ptype, pos))
                if len(patch_buf_tensors_s) >= PATCH_BATCH:
                    flush_patch_batch_stripe()
            else:
                patch_buf_tensors_g.append(t)
                patch_buf_meta_g.append((img_id, x1, y1, x2, y2, win, ptype, pos))
                if len(patch_buf_tensors_g) >= PATCH_BATCH:
                    flush_patch_batch_grid()

        if idx % 500 == 0:
            gs = sum(x.shape[0] for x in global_feats_chunks) + len(img_batch_paths)
            ps = sum(x.shape[0] for x in patch_feats_chunks_s) + len(patch_buf_tensors_s)
            pg = sum(x.shape[0] for x in patch_feats_chunks_g) + len(patch_buf_tensors_g)
            print(f"[SCAN] {idx}/{len(valid_images)} global={gs} stripe={ps} grid={pg}")

    flush_global_batch()
    flush_patch_batch_stripe()
    flush_patch_batch_grid()

    if not global_feats_chunks:
        raise RuntimeError("No global feats extracted.")

    global_feats = np.concatenate(global_feats_chunks, axis=0).astype("float32")
    img_paths = np.array(img_paths_kept, dtype=object)

    # Patch arrays
    patch_feats_s = np.concatenate(patch_feats_chunks_s, axis=0).astype("float32") if patch_feats_chunks_s else np.zeros((0, global_feats.shape[1]), np.float32)
    patch_meta_s  = np.array(patch_meta_list_s, dtype=np.int32) if patch_meta_list_s else np.zeros((0, 8), np.int32)

    patch_feats_g = np.concatenate(patch_feats_chunks_g, axis=0).astype("float32") if patch_feats_chunks_g else np.zeros((0, global_feats.shape[1]), np.float32)
    patch_meta_g  = np.array(patch_meta_list_g, dtype=np.int32) if patch_meta_list_g else np.zeros((0, 8), np.int32)

    print(f"[DONE] Global feats: {global_feats.shape}, Patch stripe feats: {patch_feats_s.shape}, Patch grid feats: {patch_feats_g.shape}")

    # If some images failed global (rare), remap ids to compact ones
    # (keeps patch meta consistent with global_img_paths.npy)
    global_set = set(img_paths.tolist())
    keep_img_ids, keep_paths = [], []
    for i0, pth in enumerate(valid_images):
        if pth in global_set:
            keep_img_ids.append(i0)
            keep_paths.append(pth)

    path_to_row = {p: i for i, p in enumerate(img_paths.tolist())}
    global_feats2 = np.zeros((len(keep_paths), global_feats.shape[1]), dtype=np.float32)
    for new_i, pth in enumerate(keep_paths):
        global_feats2[new_i] = global_feats[path_to_row[pth]]

    oldid_to_newid = {old: new for new, old in enumerate(keep_img_ids)}
    img_paths_compact = np.array(keep_paths, dtype=object)

    # filter+remap patch meta
    if len(patch_meta_s) > 0:
        keep_mask_s = np.array([int(m[0]) in oldid_to_newid for m in patch_meta_s], dtype=bool)
        patch_feats_s = patch_feats_s[keep_mask_s]
        patch_meta_s = patch_meta_s[keep_mask_s]
        patch_meta_s[:, 0] = np.array([oldid_to_newid[int(x)] for x in patch_meta_s[:, 0]], dtype=np.int32)

    if len(patch_meta_g) > 0:
        keep_mask_g = np.array([int(m[0]) in oldid_to_newid for m in patch_meta_g], dtype=bool)
        patch_feats_g = patch_feats_g[keep_mask_g]
        patch_meta_g = patch_meta_g[keep_mask_g]
        patch_meta_g[:, 0] = np.array([oldid_to_newid[int(x)] for x in patch_meta_g[:, 0]], dtype=np.int32)

    Nimg, D = global_feats2.shape
    print(f"[FINAL] Global feats: {global_feats2.shape}, Stripe feats: {patch_feats_s.shape}, Grid feats: {patch_feats_g.shape}")

    # ---------- Build GLOBAL index ----------
    if Nimg < GLOBAL_FLAT_THRESHOLD:
        print(f"[GLOBAL] FlatIP (N={Nimg})")
        g_index = faiss.IndexFlatIP(D)
        g_index.add(global_feats2)
    else:
        g_index = build_ivfflat_ip(global_feats2, seed=IVF_SEED)

    # ---------- Build PATCH indexes ----------
    if len(patch_feats_s) < PATCH_FLAT_THRESHOLD_VECS:
        print(f"[PATCH-STRIPE] FlatIP (N={len(patch_feats_s)})")
        p_index_s = faiss.IndexFlatIP(D)
        p_index_s.add(patch_feats_s)
    else:
        print(f"[PATCH-STRIPE] IVF-PQ (N={len(patch_feats_s)})")
        p_index_s = build_ivfpq_ip(patch_feats_s, seed=IVF_SEED)

    if len(patch_feats_g) < PATCH_FLAT_THRESHOLD_VECS:
        print(f"[PATCH-GRID] FlatIP (N={len(patch_feats_g)})")
        p_index_g = faiss.IndexFlatIP(D)
        p_index_g.add(patch_feats_g)
    else:
        print(f"[PATCH-GRID] IVF-PQ (N={len(patch_feats_g)})")
        p_index_g = build_ivfpq_ip(patch_feats_g, seed=IVF_SEED)

    # Save
    faiss.write_index(g_index, GLOBAL_INDEX)
    np.save(GLOBAL_META, img_paths_compact, allow_pickle=True)

    faiss.write_index(p_index_s, PATCH_STRIPE_INDEX)
    np.save(PATCH_STRIPE_META, patch_meta_s, allow_pickle=True)

    faiss.write_index(p_index_g, PATCH_GRID_INDEX)
    np.save(PATCH_GRID_META, patch_meta_g, allow_pickle=True)

    print(f"[SAVE] Global index -> {GLOBAL_INDEX}")
    print(f"[SAVE] Global paths -> {GLOBAL_META}")
    print(f"[SAVE] Stripe index -> {PATCH_STRIPE_INDEX}")
    print(f"[SAVE] Stripe meta  -> {PATCH_STRIPE_META}")
    print(f"[SAVE] Grid index   -> {PATCH_GRID_INDEX}")
    print(f"[SAVE] Grid meta    -> {PATCH_GRID_META}")
    print("[OK] DINOv3 hybrid build finished.")

if __name__ == "__main__":
    main()
