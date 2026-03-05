import os
import json
from pathlib import Path
from typing import List

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

# =========================
# 你的数据根目录
# =========================
DATASET_ROOTS = [
    r"D:\zhanlan\poptnc.com",
    r"D:\zhanlan\new_data",
    r"D:\印花",
]

# =========================
# 你需要改的：dinov3 仓库路径与权重
# =========================
REPO_DIR = r"YOUR_LOCAL_PATH_TO\dinov3"  # e.g. r"D:\repos\dinov3"
WEIGHTS = r"YOUR_WEIGHTS_PATH_OR_URL"    # e.g. r"D:\models\dinov3_vitb16.pth"

OUT_DIR = r"D:\dinov3_retrieval_out_stream"

MODEL_NAME = "dinov3_vitb16"
SCALES = (384, 512)          # 手机拍摄推荐
USE_LAST_N_LAYERS = 1        # 1=最后一层；不稳再改 2
FUSE_ALPHA = 0.5             # 0.5*CLS + 0.5*GeM
GEM_P = 3.0
BATCH_SIZE = 16              # 4060 8G 推荐从 16 开始
MAX_LONG_SIDE = 1600         # 库图很大：先限幅再做 384/512

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(roots: List[str]) -> List[str]:
    paths = []
    for r in roots:
        rp = Path(r)
        if not rp.exists():
            print(f"[WARN] root not exists: {r}")
            continue
        for p in rp.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(str(p.resolve()))
    paths.sort()
    return paths


def resize_long_side(img: Image.Image, max_long_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def safe_open_image(path: str) -> Image.Image | None:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)     # 处理手机 EXIF 旋转
        img = img.convert("RGB")
        img = resize_long_side(img, MAX_LONG_SIDE)
        return img
    except Exception:
        return None


def make_transform(size: int):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def gem_pool(feat_map: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    # [B, D, H, W] -> [B, D]
    x = feat_map.clamp(min=eps).pow(p)
    x = x.mean(dim=(-1, -2)).pow(1.0 / p)
    return x


@torch.inference_mode()
def extract_batch_embeddings(model, images: List[Image.Image], device: str) -> torch.Tensor:
    embs = []
    for size in SCALES:
        tfm = make_transform(size)
        x = torch.stack([tfm(im) for im in images], dim=0).to(device)

        # 4060：FP16 更合适
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            outs = model.get_intermediate_layers(
                x, n=USE_LAST_N_LAYERS, reshape=True, return_class_token=True, norm=True
            )

        patch_maps, clss = [], []
        for (pm, cls) in outs:
            patch_maps.append(pm)  # [B,D,H,W]
            clss.append(cls)       # [B,D]
        patch_map = torch.stack(patch_maps, dim=0).mean(dim=0)
        cls = torch.stack(clss, dim=0).mean(dim=0)

        patch_vec = gem_pool(patch_map, p=GEM_P)
        emb = FUSE_ALPHA * cls + (1.0 - FUSE_ALPHA) * patch_vec
        emb = F.normalize(emb, p=2, dim=-1)
        embs.append(emb)

    emb_final = torch.stack(embs, dim=0).mean(dim=0)
    emb_final = F.normalize(emb_final, p=2, dim=-1)
    return emb_final.float().cpu()  # [B, D] float32 cpu


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 扫描路径（得到 N_total，用于预分配 memmap）
    all_paths = list_images(DATASET_ROOTS)
    N_total = len(all_paths)
    print(f"[INFO] Found images: {N_total}")
    if N_total == 0:
        print("[ERROR] No images found. Check DATASET_ROOTS.")
        return

    # 2) 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model = torch.hub.load(REPO_DIR, MODEL_NAME, source="local", weights=WEIGHTS).to(device).eval()
    print("[INFO] Model loaded.")

    # 3) 先跑 1 张图，确定 embedding 维度 D
    probe_img = None
    probe_path = None
    for p in all_paths:
        img = safe_open_image(p)
        if img is not None:
            probe_img = img
            probe_path = p
            break
    if probe_img is None:
        print("[ERROR] All images failed to open.")
        return

    probe_emb = extract_batch_embeddings(model, [probe_img], device=device)  # [1,D]
    D = int(probe_emb.shape[1])
    print(f"[INFO] Embedding dim D = {D} (probe: {probe_path})")

    # 4) 创建 memmap .npy（预分配 N_total×D）
    emb_path = os.path.join(OUT_DIR, "embeddings.npy")
    emb_mm = open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(N_total, D))
    print(f"[INFO] Created memmap: {emb_path} shape=({N_total},{D})")

    # 5) 流式写入
    paths_jsonl = os.path.join(OUT_DIR, "paths.jsonl")
    bad_paths_txt = os.path.join(OUT_DIR, "bad_paths.txt")
    valid_count_txt = os.path.join(OUT_DIR, "valid_count.txt")

    write_idx = 0
    bad = 0

    batch_imgs, batch_src_paths = [], []

    with open(paths_jsonl, "w", encoding="utf-8") as fpaths, open(bad_paths_txt, "w", encoding="utf-8") as fbad:
        for i, p in enumerate(all_paths):
            img = safe_open_image(p)
            if img is None:
                bad += 1
                fbad.write(p + "\n")
                continue

            batch_imgs.append(img)
            batch_src_paths.append(p)

            if len(batch_imgs) == BATCH_SIZE:
                embs = extract_batch_embeddings(model, batch_imgs, device=device).numpy()  # [B,D]
                bsz = embs.shape[0]

                emb_mm[write_idx:write_idx + bsz] = embs
                for bp in batch_src_paths:
                    fpaths.write(json.dumps({"path": bp}, ensure_ascii=False) + "\n")

                write_idx += bsz
                batch_imgs.clear()
                batch_src_paths.clear()

                if (i + 1) % (BATCH_SIZE * 20) == 0:
                    print(f"[INFO] scanned={i+1}/{N_total}, written={write_idx}, bad={bad}")

        # last batch
        if batch_imgs:
            embs = extract_batch_embeddings(model, batch_imgs, device=device).numpy()
            bsz = embs.shape[0]
            emb_mm[write_idx:write_idx + bsz] = embs
            for bp in batch_src_paths:
                fpaths.write(json.dumps({"path": bp}, ensure_ascii=False) + "\n")
            write_idx += bsz

    # 6) flush & 记录有效数量
    emb_mm.flush()
    with open(valid_count_txt, "w", encoding="utf-8") as f:
        f.write(str(write_idx))

    print(f"[INFO] Done. written(valid)={write_idx}, bad={bad}")
    print(f"[INFO] embeddings.npy (allocated) rows={N_total}, use first valid rows={write_idx}")
    print(f"[INFO] Saved: {paths_jsonl}")
    print(f"[INFO] Saved: {bad_paths_txt}")
    print(f"[INFO] Saved: {valid_count_txt}")


if __name__ == "__main__":
    main()