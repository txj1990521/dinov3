import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from dinov3.models.vision_transformer import vit_large

# =========================
# CONFIG
# =========================
CKPT = r"D:/zhanlanProject/dinov3/pre_model/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DATA_ROOT = r"D:\zhanlan\new_data_noCrop"
OUT_DIR = r"D:\zhanlan\faiss_dinov3_noCrop"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_INDEX = os.path.join(OUT_DIR, "global.index")
GLOBAL_META  = os.path.join(OUT_DIR, "global_img_paths.npy")

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

# =========================
# Model
# =========================
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

def build_model():
    print("Loading DINOv3 ViT-L/16...")

    model = vit_large(patch_size=16)
    state_dict = torch.load(CKPT, map_location="cpu")

    # 官方权重通常在 key: "model"
    if "model" in state_dict:
        state_dict = state_dict["model"]

    model.load_state_dict(state_dict, strict=False)

    model.eval().to(DEVICE)
    return model

# =========================
# Utils
# =========================
def imread_unicode(p):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def list_images(root):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    files.sort()
    return files

def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # DINO 默认 normalize
    img = np.transpose(img, (2,0,1))
    return torch.from_numpy(img)

@torch.no_grad()
def extract_embedding(model, img_tensor):
    feats = model.forward_features(img_tensor)

    # case 1: forward_features returns dict (common in DINO/DINOv2/DINOv3 repos)
    if isinstance(feats, dict):
        # 优先：直接给了 cls token（很多实现提供）
        for k in ["x_norm_clstoken", "cls_token", "clstoken", "x_cls", "feat_cls"]:
            if k in feats and torch.is_tensor(feats[k]):
                cls = feats[k]
                return F.normalize(cls, dim=1)

        # 次选：给了 token 序列（N, T, C）
        for k in ["x_norm", "tokens", "x", "last_hidden_state", "x_prenorm"]:
            if k in feats and torch.is_tensor(feats[k]):
                tok = feats[k]
                # tok: [B, T, C]
                if tok.dim() == 3:
                    cls = tok[:, 0, :]
                    return F.normalize(cls, dim=1)
                # 如果是 [B, C] 就当作全局特征
                if tok.dim() == 2:
                    return F.normalize(tok, dim=1)

        raise KeyError(f"forward_features returned dict but no known keys found. keys={list(feats.keys())}")

    # case 2: forward_features returns tensor directly
    if torch.is_tensor(feats):
        if feats.dim() == 3:
            cls = feats[:, 0, :]
        elif feats.dim() == 2:
            cls = feats
        else:
            raise ValueError(f"Unexpected feats tensor shape: {feats.shape}")
        return F.normalize(cls, dim=1)

    raise TypeError(f"Unexpected forward_features type: {type(feats)}")


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model = build_dinov3_vitl16(CKPT,DEVICE)

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

    # last
    if batch:
        bt = torch.stack(batch).to(DEVICE)
        feats = extract_embedding(model, bt).cpu().numpy()
        feats_all.append(feats)
        img_paths.extend(batch_paths)

    feats_all = np.concatenate(feats_all, axis=0).astype("float32")

    print("Feature shape:", feats_all.shape)

    # =========================
    # Build FAISS
    # =========================
    D = feats_all.shape[1]
    index = faiss.IndexFlatIP(D)
    index.add(feats_all)

    faiss.write_index(index, GLOBAL_INDEX)
    np.save(GLOBAL_META, np.array(img_paths, dtype=object))

    print("Index saved.")

if __name__ == "__main__":
    main()
