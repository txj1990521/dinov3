#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid search for the DINOv3 hybrid FAISS database built by your build script.

改造点：
- 不用命令行 argparse
- 顶部 CONFIG 配置区直接改参数即可运行
"""

import os
from typing import Dict, Tuple, List

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss

# ============================================================
# CONFIG：只改这里（所有运行参数都集中在这里）
# ============================================================
CONFIG = {
    # ---- 必填：模型 ckpt + 数据库目录 + 查询图 ----
    "CKPT": r"pre_model/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "DB_DIR": r"D:\zhanlan\faiss_database_dinov3_hybrid",
    "QUERY_IMG": r"D:\zhanlan\qurrey_data\333.jpg",

    # ---- 检索参数 ----
    "TOPK": 12,

    # patch 检索：每个 query patch 在 FAISS 里取多少个近邻
    "K_PATCH": 2000,
    # 每个 query patch 实际用于聚合到 img 的 topN 命中（越大越慢）
    "TOP_PER_QUERY_PATCH": 200,

    # ---- 融合权重 ----
    "WG": 1.0,   # global
    "WS": 0.7,   # stripe patch best
    "WQ": 0.7,   # grid patch best

    # ---- IVF 的 nprobe（Flat 索引会忽略）----
    "NPROBE_GLOBAL": 64,
    "NPROBE_PATCH": 64,

    # ---- 输出更多调试信息（比如bbox）----
    "PRINT_BBOX": True,
    # ---- 结果图输出 ----
    "SAVE_MONTAGE": True,
    "MONTAGE_OUT": r"D:\zhanlan\qurrey_data\result_montage.png",

    # 网格：每行多少列（截图效果一般用 4）
    "GRID_COLS": 4,

    # 单格大小（像素）
    "CELL_W": 320,
    "CELL_H": 320,

    # 单格内边距
    "PAD": 14,

}

# ============================================================
# 运行设备
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# DINOv3 preprocess
# -------------------------
DINO_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
DINO_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)

# global view config
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
PATCH_ENC_SIZE = 224

# patch tiling config
STRIPE_AR_THR = 2.5
STRIPE_MAX_PATCHES = 24
STRIPE_WIN_H = 224
STRIPE_STRIDE = 48
STRIPE_CENTER_FRAC = 0.92
STRIPE_JITTER = 8

PATCH_SIZES = (224, 320, 384, 448)
STRIDE_RATIO = 0.5
MAX_LONG = 1024
MAX_PATCHES_PER_IMAGE = 64


# -------------------------
# utils: io
# -------------------------
def imread_unicode(p: str):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def resize_long_edge(img_bgr: np.ndarray, max_long=1024):
    h, w = img_bgr.shape[:2]
    long_ = max(h, w)
    if long_ <= max_long:
        return img_bgr
    scale = max_long / long_
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)


# -------------------------
# preprocess & aug
# -------------------------
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
        y = np.random.randint(0, h - size + 1)
        x = np.random.randint(0, w - size + 1)
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

def to_tensor_from_rgb_dino(img_rgb: np.ndarray):
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - DINO_MEAN) / DINO_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

def power_norm_torch(x: torch.Tensor, eps: float = 1e-12):
    return torch.sign(x) * torch.sqrt(torch.clamp(torch.abs(x), min=eps))


# -------------------------
# patch windows
# -------------------------
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
        hh = min(win_w, Hu)
        ww = min(win_h, W)
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

def gen_patch_windows_unified(img_bgr: np.ndarray, seed: int):
    img = resize_long_edge(img_bgr, max_long=MAX_LONG)
    H, W = img.shape[:2]
    ar = max(W / (H + 1e-6), H / (W + 1e-6))
    is_stripe = (ar >= STRIPE_AR_THR)

    windows = []
    if is_stripe:
        win_w = int(np.clip(0.85 * min(H, W), 160, 256))
        windows = gen_stripe_windows(
            H, W,
            max_patches=STRIPE_MAX_PATCHES,
            seed=seed,
            win_w=win_w,
            win_h=min(STRIPE_WIN_H, max(H, W)),
            stride=max(32, STRIPE_STRIDE),
            center_frac=STRIPE_CENTER_FRAC,
            jitter=STRIPE_JITTER,
        )
        if len(windows) < 6:
            ww = min(max(win_w, 160), W)
            hh = min(max(STRIPE_WIN_H, 256), H)
            x1 = max(0, (W - ww) // 2)
            y1 = max(0, (H - hh) // 2)
            windows.append((x1, y1, x1 + ww, y1 + hh, int(max(ww, hh)), 1, 5000))
    else:
        for win in PATCH_SIZES:
            if H < win or W < win:
                continue
            stride = max(1, int(round(win * STRIDE_RATIO)))
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
        if MAX_PATCHES_PER_IMAGE and len(windows) > MAX_PATCHES_PER_IMAGE:
            windows = sample_windows_deterministic(windows, MAX_PATCHES_PER_IMAGE, seed=seed)

    return img, windows

def patch_to_tensor(patch_bgr: np.ndarray):
    rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (PATCH_ENC_SIZE, PATCH_ENC_SIZE), interpolation=cv2.INTER_LINEAR)
    return to_tensor_from_rgb_dino(rgb)


# -------------------------
# DINOv3 model loading
# -------------------------
@torch.no_grad()
def build_dinov3_vitl16(ckpt_path: str, device: str):
    from dinov3.models.vision_transformer import vit_large

    model = vit_large(patch_size=16)

    # 建议：更安全（如果你的 torch 支持）
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

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


# -------------------------
# embedding extraction
# （用更鲁棒的版本，避免 forward_features 返回结构变化导致 None）
# -------------------------
@torch.no_grad()
def extract_dino_embedding(model, batch: torch.Tensor) -> torch.Tensor:
    feats = model.forward_features(batch)
    if feats is None:
        raise RuntimeError(f"forward_features returned None. batch={tuple(batch.shape)}")

    if isinstance(feats, dict):
        for k in ("x_norm_clstoken", "cls", "clstoken", "x", "tokens"):
            if k in feats and feats[k] is not None:
                feats = feats[k]
                break
        else:
            vals = [v for v in feats.values() if v is not None]
            if not vals:
                raise RuntimeError(f"forward_features dict has no usable values. keys={list(feats.keys())}")
            feats = vals[-1]

    if isinstance(feats, (tuple, list)):
        vals = [v for v in feats if v is not None]
        if not vals:
            raise RuntimeError("forward_features returned tuple/list but all elements are None")
        feats = vals[-1]

    if not torch.is_tensor(feats):
        raise RuntimeError(f"forward_features returned unsupported type: {type(feats)}")

    if feats.dim() == 3:
        out = feats[:, 0, :]  # CLS
    elif feats.dim() == 2:
        out = feats
    elif feats.dim() == 4:
        out = feats.mean(dim=(2, 3))
    else:
        raise RuntimeError(f"Unsupported feature shape: {tuple(feats.shape)}")

    return F.normalize(out, p=2, dim=1)


# -------------------------
# Query embeddings
# -------------------------
@torch.no_grad()
def make_views_for_global_query(img_bgr: np.ndarray, seed: int = 0) -> List[torch.Tensor]:
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
def global_query_embedding(model, img_bgr: np.ndarray) -> np.ndarray:
    views = make_views_for_global_query(img_bgr, seed=0)
    bt = torch.stack(views, dim=0).to(DEVICE)
    feats = extract_dino_embedding(model, bt)          # (V, D)
    agg = feats.max(dim=0).values                     # (D,)
    agg = power_norm_torch(agg)
    agg = F.normalize(agg.unsqueeze(0), p=2, dim=1)    # (1, D)
    return agg.cpu().numpy().astype("float32")

@torch.no_grad()
def patch_query_embeddings(model, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = 999
    img_resized, windows = gen_patch_windows_unified(img_bgr, seed=seed)

    stripe_ts, stripe_meta = [], []
    grid_ts, grid_meta = [], []

    for (x1, y1, x2, y2, win, ptype, pos) in windows:
        patch = img_resized[y1:y2, x1:x2]
        t = patch_to_tensor(patch)
        if ptype == 1:
            stripe_ts.append(t)
            stripe_meta.append((x1, y1, x2, y2, win, ptype, pos))
        else:
            grid_ts.append(t)
            grid_meta.append((x1, y1, x2, y2, win, ptype, pos))

    def encode(ts_list):
        if not ts_list:
            return np.zeros((0, 1), dtype=np.float32)
        bt = torch.stack(ts_list, dim=0).to(DEVICE)
        feats = extract_dino_embedding(model, bt).cpu().numpy().astype("float32")
        return feats

    feats_s = encode(stripe_ts)
    feats_g = encode(grid_ts)
    return feats_s, np.array(stripe_meta, dtype=np.int32), feats_g, np.array(grid_meta, dtype=np.int32)


# -------------------------
# Patch result aggregation
# -------------------------
def best_patch_scores_per_image(
    D: np.ndarray,
    I: np.ndarray,
    patch_meta: np.ndarray,
    top_per_query: int,
) -> Tuple[Dict[int, float], Dict[int, Tuple[int,int,int,int]]]:
    best_score: Dict[int, float] = {}
    best_bbox: Dict[int, Tuple[int,int,int,int]] = {}

    Q, K = I.shape
    K = min(K, top_per_query)

    for qi in range(Q):
        for r in range(K):
            pid = int(I[qi, r])
            if pid < 0:
                continue
            score = float(D[qi, r])
            img_id = int(patch_meta[pid, 0])
            if (img_id not in best_score) or (score > best_score[img_id]):
                best_score[img_id] = score
                x1, y1, x2, y2 = patch_meta[pid, 1:5].tolist()
                best_bbox[img_id] = (x1, y1, x2, y2)

    return best_score, best_bbox

def _fit_to_cell(img_bgr: np.ndarray, cell_w: int, cell_h: int, pad: int) -> np.ndarray:
    """按比例缩放并居中贴到 cell（黑底），保留纵横比。"""
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    if img_bgr is None:
        return canvas

    H, W = img_bgr.shape[:2]
    max_w = max(1, cell_w - 2 * pad)
    max_h = max(1, cell_h - 2 * pad)

    scale = min(max_w / (W + 1e-6), max_h / (H + 1e-6))
    nw = max(1, int(round(W * scale)))
    nh = max(1, int(round(H * scale)))

    img_rs = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y1 = pad + (max_h - nh) // 2
    x1 = pad + (max_w - nw) // 2
    canvas[y1:y1+nh, x1:x1+nw] = img_rs
    return canvas


def save_montage(
    query_bgr: np.ndarray,
    fused_list: List[Tuple[float, int, float, float, float]],
    img_paths: List[str],
    out_path: str,
    cols: int = 4,
    cell_w: int = 320,
    cell_h: int = 320,
    pad: int = 14,
):
    """
    生成类似你截图的结果图：
    - 第一个格子：QUERY
    - 后面依次 #1..#TOPK：显示 fused 分数（你也可以改成 global/stripe/grid）
    """
    # 需要的格子数量：1(QUERY) + topk
    topk = len(fused_list)
    total = 1 + topk
    rows = int(np.ceil(total / cols))

    # 大画布
    H = rows * cell_h
    W = cols * cell_w
    big = np.zeros((H, W, 3), dtype=np.uint8)

    def put_label(img, text, x, y):
        # 白字 + 黑色描边（更清晰）
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # ---- 填 QUERY ----
    cell0 = _fit_to_cell(query_bgr, cell_w, cell_h, pad)
    put_label(cell0, "QUERY", 10, 34)
    big[0:cell_h, 0:cell_w] = cell0

    # ---- 填 TopK ----
    for i, (sf, img_id, sg, ss, sq) in enumerate(fused_list, start=1):
        pos = i  # 因为 0 是 query
        r = pos // cols
        c = pos % cols

        x0 = c * cell_w
        y0 = r * cell_h

        path = img_paths[img_id] if 0 <= img_id < len(img_paths) else None
        img = imread_unicode(path) if path else None

        cell = _fit_to_cell(img, cell_w, cell_h, pad)
        put_label(cell, f"#{i}  {sf:.3f}", 10, 34)

        big[y0:y0+cell_h, x0:x0+cell_w] = cell

    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ok = cv2.imencode(".png", big)[1].tofile(out_path)
    return ok

# -------------------------
# Main
# -------------------------
def main():
    ckpt = CONFIG["CKPT"]
    db_dir = CONFIG["DB_DIR"]
    query_path = CONFIG["QUERY_IMG"]

    topk = int(CONFIG["TOPK"])
    k_patch = int(CONFIG["K_PATCH"])
    top_per_query = int(CONFIG["TOP_PER_QUERY_PATCH"])

    wg = float(CONFIG["WG"])
    ws = float(CONFIG["WS"])
    wq = float(CONFIG["WQ"])

    nprobe_global = int(CONFIG["NPROBE_GLOBAL"])
    nprobe_patch = int(CONFIG["NPROBE_PATCH"])
    print_bbox = bool(CONFIG["PRINT_BBOX"])

    # db paths
    g_index_path = os.path.join(db_dir, "global.index")
    g_meta_path  = os.path.join(db_dir, "global_img_paths.npy")

    ps_index_path = os.path.join(db_dir, "patch_stripe.index")
    pg_index_path = os.path.join(db_dir, "patch_grid.index")
    ps_meta_path  = os.path.join(db_dir, "patch_stripe_meta.npy")
    pg_meta_path  = os.path.join(db_dir, "patch_grid_meta.npy")

    print("[INFO] device:", DEVICE)
    print("[INFO] ckpt:", ckpt)
    print("[INFO] db_dir:", db_dir)
    print("[INFO] query:", query_path)

    # load model
    model = build_dinov3_vitl16(ckpt, DEVICE)

    # load db
    g_index = faiss.read_index(g_index_path)
    img_paths = np.load(g_meta_path, allow_pickle=True).tolist()

    ps_index = faiss.read_index(ps_index_path)
    pg_index = faiss.read_index(pg_index_path)
    ps_meta = np.load(ps_meta_path, allow_pickle=True)
    pg_meta = np.load(pg_meta_path, allow_pickle=True)

    # set nprobe if IVF
    if hasattr(g_index, "nprobe"):
        g_index.nprobe = min(nprobe_global, getattr(g_index, "nlist", nprobe_global))
    if hasattr(ps_index, "nprobe"):
        ps_index.nprobe = min(nprobe_patch, getattr(ps_index, "nlist", nprobe_patch))
    if hasattr(pg_index, "nprobe"):
        pg_index.nprobe = min(nprobe_patch, getattr(pg_index, "nlist", nprobe_patch))

    # read query
    qimg = imread_unicode(query_path)
    if qimg is None:
        raise RuntimeError(f"Cannot read query image: {query_path}")

    # query embeddings
    qg = global_query_embedding(model, qimg)  # (1, D)
    qs, _, qk, _ = patch_query_embeddings(model, qimg)

    # global search: 多取一些候选用于融合
    Dg, Ig = g_index.search(qg, max(topk * 50, 200))
    Dg = Dg[0]
    Ig = Ig[0]
    global_score: Dict[int, float] = {int(Ig[i]): float(Dg[i]) for i in range(len(Ig)) if int(Ig[i]) >= 0}

    # patch search + aggregate best per image
    stripe_best, stripe_bbox = {}, {}
    grid_best, grid_bbox = {}, {}

    if qs.shape[0] > 0 and ps_index.ntotal > 0:
        Ds, Is = ps_index.search(qs, k_patch)
        stripe_best, stripe_bbox = best_patch_scores_per_image(Ds, Is, ps_meta, top_per_query)

    if qk.shape[0] > 0 and pg_index.ntotal > 0:
        Dk, Ik = pg_index.search(qk, k_patch)
        grid_best, grid_bbox = best_patch_scores_per_image(Dk, Ik, pg_meta, top_per_query)

    # fuse
    cand = set(global_score.keys()) | set(stripe_best.keys()) | set(grid_best.keys())

    def get(d, k):
        return d.get(k, 0.0)

    fused = []
    for img_id in cand:
        sg = get(global_score, img_id)
        ss = get(stripe_best, img_id)
        sq = get(grid_best, img_id)
        sf = wg * sg + ws * ss + wq * sq
        fused.append((sf, img_id, sg, ss, sq))

    fused.sort(reverse=True, key=lambda x: x[0])
    fused = fused[:topk]
    # ---- 保存结果拼图 ----
    if CONFIG.get("SAVE_MONTAGE", True):
        out_path = CONFIG["MONTAGE_OUT"]
        cols = int(CONFIG.get("GRID_COLS", 4))
        cell_w = int(CONFIG.get("CELL_W", 320))
        cell_h = int(CONFIG.get("CELL_H", 320))
        pad = int(CONFIG.get("PAD", 14))

        save_montage(
            query_bgr=qimg,
            fused_list=fused,          # 注意：这里 fused 里已经是 topk
            img_paths=img_paths,
            out_path=out_path,
            cols=cols,
            cell_w=cell_w,
            cell_h=cell_h,
            pad=pad,
        )
        print(f"[SAVE] montage -> {out_path}")

    # print
    print("=" * 80)
    print("QUERY:", query_path)
    print(f"TOPK={topk}  weights: wg={wg} ws={ws} wq={wq}")
    print("=" * 80)

    for rank, (sf, img_id, sg, ss, sq) in enumerate(fused, 1):
        path = img_paths[img_id] if 0 <= img_id < len(img_paths) else "<out-of-range>"
        print(f"[{rank:02d}] fused={sf:.4f}  global={sg:.4f}  stripe={ss:.4f}  grid={sq:.4f}")
        print(f"     img_id={img_id}  path={path}")

        if print_bbox:
            sb = stripe_bbox.get(img_id, None)
            gb = grid_bbox.get(img_id, None)
            if sb is not None:
                print(f"     best_stripe_bbox={sb}")
            if gb is not None:
                print(f"     best_grid_bbox={gb}")

if __name__ == "__main__":
    main()
