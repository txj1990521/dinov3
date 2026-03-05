# build_index_dinov3_registered.py
# -*- coding: utf-8 -*-
"""
DINOv3 Index Builder with Registry Alignment.
Ensures vector indices match exactly with the CLIP builder for fusion search.
"""
import os, json, time, hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
from PIL import Image
import faiss
import torch
from transformers import AutoModel, AutoImageProcessor

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ============================================================
# ✅ CONFIG
# ============================================================
CONFIG: Dict[str, Any] = {
    # 【关键】必须指向 gen_registry.py 生成的同一个文件
    "REGISTRY_PATH": r".\outputs_hybrid_folder_big\image_registry.json",

    # 模型配置 (DINOv3)
    # 可选: facebook/dinov3-vits16-pretrain-lvd1689m (Small)
    #      facebook/dinov3-vitb16-pretrain-lvd1689m (Base, 推荐)
    #      facebook/dinov3-vitl16-pretrain-lvd1689m (Large)
    "DINO_MODEL_ID": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "DEVICE": "auto",

    # 输出目录 (建议与 CLIP 保持一致或同级)
    "OUT_DIR": r".\outputs_hybrid_folder_big",

    # --- 策略配置 (与 CLIP 脚本保持逻辑一致) ---
    "GLOBAL_USE_5CROP": True,
    "GLOBAL_CROP_RATIO": 0.8,

    "PATCH_ENABLE": True,
    "PATCH_RESIZE_LONG": 1600,  # 长边限制，防止显存爆炸
    "PATCH_BATCH": 256,  # Patch 推理批大小

    # Stripe 参数
    "STRIPE_AR_THR": 2.4,
    "STRIPE_MAX_PATCHES": 12,
    "STRIPE_WIN_H": 336,  # DINOv3 通常输入 336 或 518
    "STRIPE_STRIDE": 96,
    "STRIPE_CENTER_FRAC": 0.92,
    "STRIPE_JITTER": 8,
    "STRIPE_WIN_W_FRAC": 0.85,
    "STRIPE_WIN_W_MIN": 160,
    "STRIPE_WIN_W_MAX": 256,

    # Grid 参数
    "GRID_SIZES": [512, 384, 256],
    "GRID_STRIDE_RATIO": 0.5,
    "GRID_MAX_PATCHES": 12,

    # --- 输出文件名 (加 dinov3 前缀以示区分，但结构一致) ---
    "OUT_GLOBAL_INDEX": "dinov3_global.index",
    "OUT_PATCH_INDEX": "dinov3_patch.index",
    "OUT_IMAGES_META": "dinov3_images_meta.json",
    "OUT_GLOBAL_META": "dinov3_global_meta.jsonl",
    "OUT_PATCH_META": "dinov3_patch_meta.jsonl",
    "OUT_G_VEC2IMG": "dinov3_g_vec_to_img.npy",
    "OUT_P_VEC2IMG": "dinov3_p_vec_to_img.npy",
    "OUT_P_BBOX": "dinov3_p_vec_to_bbox.npy",
    "BAD_LOG": "dinov3_bad_paths.log",

    # --- FAISS 索引配置 ---
    # 维度由模型决定 (ViT-B: 768, ViT-L: 1024)
    "GLOBAL_INDEX_TYPE": "IVF_FLAT",
    "GLOBAL_NLIST": 1024,
    "PATCH_INDEX_TYPE": "IVF_PQ",
    "PATCH_NLIST": 4096,
    "PQ_M": 64,  # 需能被维度整除 (768/64=12, 1024/64=16)
    "PQ_NBITS": 8,

    # --- 训练采样配置 ---
    "TRAIN_MAX_IMAGES": 5000,
    "TRAIN_MAX_GLOBAL_VECS": 80000,
    "TRAIN_MAX_PATCH_VECS": 80000,
    "TRAIN_SEED": 123,
    "TRAIN_MIN_GLOBAL_VECS": 5000,
    "TRAIN_MIN_PATCH_VECS": 8000,

    "PROFILE_EVERY": 2000,
}


# ============================================================
# Utils
# ============================================================
def pick_device(cfg: Dict[str, Any]) -> str:
    dv = str(cfg.get("DEVICE", "auto")).lower()
    if dv in ("cuda", "gpu"): return "cuda"
    if dv == "cpu": return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def stable_seed_from_str(s: str, base: int = 0) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base) & 0x7fffffff


def load_registry(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Registry not found: {path}. Please run gen_registry.py first.")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded registry with {len(data)} images from {path}")
    return data


def five_crops(pil_img: Image.Image, crop_ratio=0.8) -> List[Image.Image]:
    w, h = pil_img.size
    s = int(min(w, h) * crop_ratio)
    if s <= 0: return []
    coords = [
        ((w - s) // 2, (h - s) // 2),
        (0, 0), (w - s, 0), (0, h - s), (w - s, h - s),
    ]
    return [pil_img.crop((x, y, x + s, y + s)) for x, y in coords]


def maybe_resize_long(pil: Image.Image, max_long: int) -> Image.Image:
    if not max_long: return pil
    w, h = pil.size
    if max(w, h) <= max_long: return pil
    scale = max_long / float(max(w, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return pil.resize((nw, nh), resample=Image.BICUBIC)


def encode_pil_batch_dino(model, processor, device, imgs: List[Image.Image]) -> np.ndarray:
    if not imgs:
        dim = model.config.hidden_size
        return np.zeros((0, dim), dtype=np.float32)

    # DINOv3 使用 transformers processor
    inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)

    use_amp = str(device).startswith("cuda")
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=use_amp):
        outputs = model(**inputs)
        # 取 CLS token (第一个 token)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        # L2 Norm
        cls_tokens = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)

    return cls_tokens.float().cpu().numpy().astype("float32")


# ============================================================
# Patch Window Logic (Same as CLIP script for consistency)
# ============================================================
def is_stripe_shape(w: int, h: int, ar_thr: float) -> bool:
    ar = max(w / (h + 1e-6), h / (w + 1e-6))
    return ar >= ar_thr


def gen_stripe_windows(W, H, max_patches, seed, win_w, win_h, stride, center_frac, jitter):
    rng = np.random.default_rng(seed)
    vertical = (H >= W)
    windows = []
    if vertical:
        Wu = int(round(W * center_frac))
        x0 = max(0, (W - Wu) // 2)
        ww = min(win_w, Wu)
        hh = min(win_h, H)
        if ww < 16 or hh < 16: return []
        x_base = x0 + max(0, (Wu - ww) // 2)
        y = 0
        while y + hh <= H and len(windows) < max_patches:
            dx = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            x1 = int(np.clip(x_base + dx, x0, x0 + Wu - ww))
            y1 = int(y)
            x2, y2 = x1 + ww, y1 + hh
            center = (y1 + y2) * 0.5
            pos = int(np.clip((center / max(1.0, H)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))
            y += stride
    else:
        Hu = int(round(H * center_frac))
        y0 = max(0, (H - Hu) // 2)
        hh = min(win_w, Hu)
        ww = min(win_h, W)
        if hh < 16 or ww < 16: return []
        y_base = y0 + max(0, (Hu - hh) // 2)
        x = 0
        while x + ww <= W and len(windows) < max_patches:
            dy = int(rng.integers(-jitter, jitter + 1)) if jitter > 0 else 0
            y1 = int(np.clip(y_base + dy, y0, y0 + Hu - hh))
            x1 = int(x)
            x2, y2 = x1 + ww, y1 + hh
            center = (x1 + x2) * 0.5
            pos = int(np.clip((center / max(1.0, W)) * 10000.0, 0, 10000))
            windows.append((x1, y1, x2, y2, int(max(ww, hh)), 1, pos))
            x += stride
    return windows


def gen_grid_windows(W, H, sizes, stride_ratio, max_patches, seed):
    windows = []
    for win in sizes:
        if H < win or W < win: continue
        stride = max(1, int(round(win * stride_ratio)))
        for y1 in range(0, H - win + 1, stride):
            for x1 in range(0, W - win + 1, stride):
                windows.append((x1, y1, x1 + win, y1 + win, win, 0, 5000))
    # Center fallback
    for win in (512, 384, 256, 224):
        if H >= win and W >= win:
            x1 = (W - win) // 2
            y1 = (H - win) // 2
            windows.append((x1, y1, x1 + win, y1 + win, win, 0, 5000))
            break

    windows = list(dict.fromkeys(windows))
    if max_patches and len(windows) > max_patches:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(windows), size=max_patches, replace=False))
        windows = [windows[i] for i in idx.tolist()]
    return windows


def gen_patch_windows_for_image(pil_img, cfg, seed):
    W, H = pil_img.size
    if is_stripe_shape(W, H, float(cfg["STRIPE_AR_THR"])):
        ww = int(np.clip(cfg["STRIPE_WIN_W_FRAC"] * min(W, H),
                         cfg["STRIPE_WIN_W_MIN"],
                         cfg["STRIPE_WIN_W_MAX"]))
        return gen_stripe_windows(
            W, H, int(cfg["STRIPE_MAX_PATCHES"]), seed,
            ww, int(cfg["STRIPE_WIN_H"]), int(cfg["STRIPE_STRIDE"]),
            float(cfg["STRIPE_CENTER_FRAC"]), int(cfg["STRIPE_JITTER"])
        )
    return gen_grid_windows(
        W, H, list(cfg["GRID_SIZES"]), float(cfg["GRID_STRIDE_RATIO"]),
        int(cfg["GRID_MAX_PATCHES"]), seed
    )


def make_index(d, kind, nlist, metric=faiss.METRIC_INNER_PRODUCT, pq_m=64, pq_nbits=8):
    kind = kind.upper()
    if kind == "FLAT": return faiss.IndexFlatIP(d)
    if kind == "IVF_FLAT": return faiss.index_factory(d, f"IVF{int(nlist)},Flat", metric)
    if kind == "IVF_PQ":
        if d % int(pq_m) != 0:
            # 自动调整 M 以适配维度
            original_m = pq_m
            pq_m = d // (d // pq_m)  # 简单启发式，或者报错
            if d % pq_m != 0:
                pq_m = d // 8  # fallback
            print(f"Warning: Adjusted PQ_M from {original_m} to {pq_m} to fit dim {d}")
        return faiss.index_factory(d, f"IVF{int(nlist)},PQ{int(pq_m)}x{int(pq_nbits)}", metric)
    raise ValueError(f"Unknown index kind: {kind}")


def maybe_take(rng, keep_prob):
    return float(rng.random()) < float(keep_prob)


# ============================================================
# Main
# ============================================================
def main():
    cfg = CONFIG
    out_dir = cfg["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Registry
    registry = load_registry(cfg["REGISTRY_PATH"])
    if not registry:
        raise RuntimeError("Registry is empty!")

    # 2. Init Model
    device = pick_device(cfg)
    print(f"Loading DINOv3 model: {cfg['DINO_MODEL_ID']} on {device}")
    processor = AutoImageProcessor.from_pretrained(cfg["DINO_MODEL_ID"])
    model = AutoModel.from_pretrained(cfg["DINO_MODEL_ID"]).to(device).eval()

    # Get Dimension dynamically
    DIM = model.config.hidden_size
    print(f"Model Dimension: {DIM}")

    patch_enable = bool(cfg.get("PATCH_ENABLE", True))
    patch_batch = int(cfg.get("PATCH_BATCH", 256))
    GLOBAL_BATCH = 64
    WRITE_FLUSH_LINES = 8000

    # 3. Paths
    global_index_path = os.path.join(out_dir, cfg["OUT_GLOBAL_INDEX"])
    patch_index_path = os.path.join(out_dir, cfg["OUT_PATCH_INDEX"])
    g_vec2img_path = os.path.join(out_dir, cfg["OUT_G_VEC2IMG"])
    p_vec2img_path = os.path.join(out_dir, cfg["OUT_P_VEC2IMG"])
    p_bbox_path = os.path.join(out_dir, cfg["OUT_P_BBOX"])
    global_meta_path = os.path.join(out_dir, cfg["OUT_GLOBAL_META"])
    patch_meta_path = os.path.join(out_dir, cfg["OUT_PATCH_META"])
    images_meta_path = os.path.join(out_dir, cfg["OUT_IMAGES_META"])
    bad_log_path = os.path.join(out_dir, cfg["BAD_LOG"])

    # 4. PASS 1: Training Sampling
    rng = np.random.default_rng(int(cfg.get("TRAIN_SEED", 123)))
    train_max_images = int(cfg.get("TRAIN_MAX_IMAGES", 5000))
    train_max_g = int(cfg.get("TRAIN_MAX_GLOBAL_VECS", 80000))
    train_max_p = int(cfg.get("TRAIN_MAX_PATCH_VECS", 80000))
    train_min_g = int(cfg.get("TRAIN_MIN_GLOBAL_VECS", 5000))
    train_min_p = int(cfg.get("TRAIN_MIN_PATCH_VECS", 8000))

    train_g, train_p = [], []
    train_g_cnt, train_p_cnt = 0, 0

    est_g_per_img = 5 if cfg.get("GLOBAL_USE_5CROP", True) else 1
    est_p_per_img = 10.0 if patch_enable else 0.0
    scan_n = min(len(registry), train_max_images)

    # Dynamic probability
    keep_prob_g = 1.0 if scan_n <= 20000 else min(1.0, train_max_g / max(1.0, scan_n * est_g_per_img))
    keep_prob_p = 1.0 if scan_n <= 20000 else min(1.0, train_max_p / max(1.0, scan_n * max(1.0, est_p_per_img)))

    print(f"[PASS1] Sampling from first {scan_n} images...")
    for i in tqdm(range(scan_n), desc="PASS1: Sample"):
        item = registry[i]
        img_path = item["abs_path"]
        try:
            pil0 = Image.open(img_path).convert("RGB")
        except:
            continue

        # Global
        crops = five_crops(pil0, float(cfg["GLOBAL_CROP_RATIO"])) if cfg.get("GLOBAL_USE_5CROP", True) else [pil0]
        if crops and train_g_cnt < train_max_g:
            selected = [c for c in crops if maybe_take(rng, keep_prob_g)]
            if selected:
                feats = l2norm(encode_pil_batch_dino(model, processor, device, selected))
                train_g.append(feats)
                train_g_cnt += feats.shape[0]

        # Patch
        if patch_enable and train_p_cnt < train_max_p:
            pil_patch = maybe_resize_long(pil0, cfg.get("PATCH_RESIZE_LONG"))
            seed = stable_seed_from_str(img_path, base=999)
            windows = gen_patch_windows_for_image(pil_patch, cfg, seed)
            patch_imgs = []
            for (x1, y1, x2, y2, win, ptype, pos) in windows:
                if train_p_cnt + len(patch_imgs) >= train_max_p: break
                if maybe_take(rng, keep_prob_p):
                    patch_imgs.append(pil_patch.crop((x1, y1, x2, y2)))
            if patch_imgs:
                feats = l2norm(encode_pil_batch_dino(model, processor, device, patch_imgs))
                train_p.append(feats)
                train_p_cnt += feats.shape[0]

        if train_g_cnt >= train_max_g and (not patch_enable or train_p_cnt >= train_max_p):
            break

    Tg = np.vstack(train_g).astype("float32") if train_g else None
    Tp = np.vstack(train_p).astype("float32") if (patch_enable and train_p) else None
    print(f"[PASS1 Done] G:{Tg.shape if Tg is not None else 0}, P:{Tp.shape if Tp is not None else 0}")

    # 5. Build & Train Index
    g_index = make_index(DIM, cfg["GLOBAL_INDEX_TYPE"], cfg["GLOBAL_NLIST"])
    p_index = None
    if patch_enable:
        # Auto-adjust PQ_M if needed inside make_index
        p_index = make_index(DIM, cfg["PATCH_INDEX_TYPE"], cfg["PATCH_NLIST"], pq_m=cfg["PQ_M"],
                             pq_nbits=cfg["PQ_NBITS"])

    if not g_index.is_trained:
        if Tg is None or Tg.shape[0] < train_min_g:
            raise RuntimeError(f"Not enough global train vectors: {Tg.shape[0] if Tg is not None else 0}")
        print("[TRAIN] Global Index...")
        g_index.train(Tg)

    if p_index and not p_index.is_trained:
        if Tp is None or Tp.shape[0] < train_min_p:
            raise RuntimeError(f"Not enough patch train vectors: {Tp.shape[0] if Tp is not None else 0}")
        print("[TRAIN] Patch Index...")
        p_index.train(Tp)

    # 6. PASS 2: Full Build (Strict Registry Order)
    gmf = open(global_meta_path, "w", encoding="utf-8", buffering=1 << 20)
    pmf = open(patch_meta_path, "w", encoding="utf-8", buffering=1 << 20) if patch_enable else None

    g_vec_to_img = []
    p_vec_to_img = []
    p_vec_to_bbox = []

    global_batch_imgs, global_batch_meta = [], []
    patch_batch_imgs, patch_batch_meta = [], []
    buf_g_lines, buf_p_lines = [], []

    stats = {"total": 0, "ok": 0, "bad": 0}
    bad_records = []

    def flush_global():
        nonlocal global_batch_imgs, global_batch_meta, buf_g_lines
        if not global_batch_imgs: return
        feats = l2norm(encode_pil_batch_dino(model, processor, device, global_batch_imgs))
        g_index.add(feats.astype("float32"))
        for m in global_batch_meta:
            buf_g_lines.append(json.dumps(m, ensure_ascii=False) + "\n")
        if len(buf_g_lines) >= WRITE_FLUSH_LINES:
            gmf.write("".join(buf_g_lines))
            buf_g_lines = []
        global_batch_imgs, global_batch_meta = [], []

    def flush_patch():
        nonlocal patch_batch_imgs, patch_batch_meta, buf_p_lines, p_vec_to_img, p_vec_to_bbox
        if not patch_enable or not p_index or not patch_batch_imgs:
            patch_batch_imgs, patch_batch_meta = [], []
            return
        feats = l2norm(encode_pil_batch_dino(model, processor, device, patch_batch_imgs))
        p_index.add(feats.astype("float32"))
        for m in patch_batch_meta:
            buf_p_lines.append(json.dumps(m, ensure_ascii=False) + "\n")
            p_vec_to_img.append(int(m["img_id"]))
            p_vec_to_bbox.append(m["bbox"])
        if len(buf_p_lines) >= WRITE_FLUSH_LINES:
            pmf.write("".join(buf_p_lines))
            buf_p_lines = []
        patch_batch_imgs, patch_batch_meta = [], []

    print(f"[PASS2] Building full index for {len(registry)} images...")
    for img_id, item in enumerate(tqdm(registry, desc="PASS2: Build")):
        stats["total"] += 1
        img_path = item["abs_path"]
        key = item["key"]

        try:
            pil0 = Image.open(img_path).convert("RGB")
        except Exception as e:
            stats["bad"] += 1
            bad_records.append(f"{img_path}: {e}")
            continue

        src_w, src_h = pil0.size

        # --- Global ---
        crops = five_crops(pil0, float(cfg["GLOBAL_CROP_RATIO"])) if cfg.get("GLOBAL_USE_5CROP", True) else [pil0]
        view_names = ["center", "tl", "tr", "bl", "br"] if cfg.get("GLOBAL_USE_5CROP", True) else ["full"]

        if not crops:
            stats["bad"] += 1
            continue

        for vi, crop_im in enumerate(crops):
            meta = {
                "vec_type": "global", "img_id": img_id, "key": key, "abs_path": img_path,
                "view": view_names[vi] if vi < len(view_names) else f"v{vi}", "src_wh": [src_w, src_h]
            }
            global_batch_imgs.append(crop_im)
            global_batch_meta.append(meta)
            g_vec_to_img.append(img_id)  # CRITICAL: Matches Registry Index

            if len(global_batch_imgs) >= GLOBAL_BATCH:
                flush_global()

        # --- Patch ---
        if patch_enable:
            pil_patch = maybe_resize_long(pil0, cfg.get("PATCH_RESIZE_LONG"))
            scale_x = pil0.size[0] / pil_patch.size[0]
            scale_y = pil0.size[1] / pil_patch.size[1]
            seed = stable_seed_from_str(img_path, base=999)
            windows = gen_patch_windows_for_image(pil_patch, cfg, seed)

            for (x1, y1, x2, y2, win, ptype, pos) in windows:
                ox1, oy1 = int(round(x1 * scale_x)), int(round(y1 * scale_y))
                ox2, oy2 = int(round(x2 * scale_x)), int(round(y2 * scale_y))
                ox1, oy1 = max(0, min(ox1, src_w - 1)), max(0, min(oy1, src_h - 1))
                ox2, oy2 = max(1, min(ox2, src_w)), max(1, min(oy2, src_h))
                if ox2 - ox1 < 8 or oy2 - oy1 < 8: continue

                patch_img = pil0.crop((ox1, oy1, ox2, oy2))
                meta = {
                    "vec_type": "patch", "img_id": img_id, "key": key, "abs_path": img_path,
                    "bbox": [ox1, oy1, ox2, oy2], "ptype": "stripe" if ptype == 1 else "grid",
                    "pos": int(pos), "win": int(win), "src_wh": [src_w, src_h]
                }
                patch_batch_imgs.append(patch_img)
                patch_batch_meta.append(meta)

                if len(patch_batch_imgs) >= patch_batch:
                    flush_patch()

        stats["ok"] += 1

    # Final Flush
    flush_global()
    if patch_enable: flush_patch()
    if buf_g_lines: gmf.write("".join(buf_g_lines))
    if patch_enable and pmf and buf_p_lines: pmf.write("".join(buf_p_lines))

    gmf.close()
    if pmf: pmf.close()

    # Save Artifacts
    with open(images_meta_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False)

    if bad_records:
        with open(bad_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(bad_records))

    np.save(g_vec2img_path, np.asarray(g_vec_to_img, dtype=np.int32))
    if patch_enable:
        np.save(p_vec2img_path, np.asarray(p_vec_to_img, dtype=np.int32))
        np.save(p_bbox_path, np.asarray(p_vec_to_bbox, dtype=np.int32))

    faiss.write_index(g_index, global_index_path)
    if patch_enable and p_index:
        faiss.write_index(p_index, patch_index_path)

    print("\n===== DINOv3 BUILD DONE (REGISTERED) =====")
    print(f"Total OK: {stats['ok']} | Bad: {stats['bad']}")
    print(f"Global Vectors: {g_index.ntotal}")
    if patch_enable: print(f"Patch Vectors: {p_index.ntotal}")
    print(f"Mapping Check: First vec img_id = {g_vec_to_img[0]}, Last vec img_id = {g_vec_to_img[-1]}")
    print(f"Output Dir: {out_dir}")


if __name__ == "__main__":
    main()