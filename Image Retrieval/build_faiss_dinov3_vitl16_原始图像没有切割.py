import os
import numpy as np
import cv2
import torch
import faiss
from pathlib import Path
from dinov3.models.vision_transformer import vit_large
import torch.nn.functional as F
# =========================
# CONFIG
# =========================
CKPT = r"D:/zhanlanProject/dinov3/pre_model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DATA_ROOT = r"D:\zhanlan\new_data_noCrop"
OUT_DIR   = r"D:\zhanlan\faiss_dinov3_L_noCrop"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_INDEX = os.path.join(OUT_DIR, "global.index")
GLOBAL_META  = os.path.join(OUT_DIR, "global_img_paths.npy")

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# =========================
# Model
# =========================
def build_model():
    print("Loading DINOv3 ViT-L/16 from Meta .pth (pure state_dict)...")

    model = vit_large(patch_size=16)

    # 你的 ckpt 是 OrderedDict（纯 state_dict），没有 ["model"]
    state_dict = torch.load(CKPT, map_location="cpu", weights_only=True)

    # 兼容有前缀的情况（你这份看起来没有，但加上不亏）
    cleaned = {}
    for k, v in state_dict.items():
        kk = k
        for pref in ("module.", "model.", "backbone."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[MODEL] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(DEVICE)
    return model

# =========================
# Utils
# =========================
def gem_pool(x, p=3.0, eps=1e-6):
    # x: [B, N, C]
    return (x.clamp(min=eps).pow(p).mean(dim=1)).pow(1.0 / p)

def imread_unicode(p):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def list_images(root):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def preprocess(img_bgr):
    if img_bgr is None:
        return None

    # 强制 3 通道
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = 224 / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

    y0 = max(0, (nh - 224) // 2)
    x0 = max(0, (nw - 224) // 2)
    img = img[y0:y0+224, x0:x0+224]
    # 兜底：保证一定是 224x224
    if img.shape[0] != 224 or img.shape[1] != 224:
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1)).astype(np.float32)

    return torch.from_numpy(img)

@torch.no_grad()
def extract_embedding(model, img_tensor):
    feats = model.forward_features(img_tensor)
    emb = feats["x_norm_clstoken"]              # [B,C]
    emb = F.normalize(emb, dim=1)
    return emb

# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = build_model()

    paths = list_images(DATA_ROOT)
    print("Found images:", len(paths))

    feats_all = []
    img_paths = []

    batch = []
    batch_paths = []

    BATCH_SIZE = 128

    for i, p in enumerate(paths):
        img = imread_unicode(str(p))
        if img is None:
            continue
        tensor = preprocess(img)
        if tensor is None or tensor.numel() == 0:
            continue
        batch.append(tensor)
        batch_paths.append(str(p))

        if len(batch) == BATCH_SIZE:
            bt = torch.stack(batch).to(DEVICE)
            feats = extract_embedding(model, bt).cpu().numpy()

            feats_all.append(feats)
            img_paths.extend(batch_paths)

            batch = []
            batch_paths = []

        if i % 500 == 0:
            print(f"{i}/{len(paths)}")

    if batch:
        bt = torch.stack(batch).to(DEVICE)
        feats = extract_embedding(model, bt).cpu().numpy()
        feats_all.append(feats)
        img_paths.extend(batch_paths)

    feats_all = np.concatenate(feats_all, axis=0).astype("float32")
    print("Feature shape:", feats_all.shape)
    D = feats_all.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(feats_all)  # feats_all 已 normalize

    faiss.write_index(index, GLOBAL_INDEX)
    np.save(GLOBAL_META, np.array(img_paths, dtype=object))

    print("Index saved.")

if __name__ == "__main__":
    main()