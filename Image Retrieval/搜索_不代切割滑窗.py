import os
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from dinov3.models.vision_transformer import vit_large
import math
import cv2
import numpy as np
# =========================
# CONFIG
# =========================
CKPT = r"D:/zhanlanProject/dinov3/pre_model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
# OUT_DIR = r"D:\zhanlan\faiss_dinov3"
OUT_DIR = r"D:\zhanlan\faiss_dinov3_L_noCrop"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_INDEX = os.path.join(OUT_DIR, "global.index")
GLOBAL_META  = os.path.join(OUT_DIR, "global_img_paths.npy")
IMAGE_SIZE=1024

# 你要搜的图片（可改成自己的路径）
QUERY_IMG = r"D:\zhanlan\qurrey_data\333.jpg"

TOPK = 10

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# =========================
# Model
# =========================
def build_dinov3_vitl16(ckpt_path: str, device: str):
    model = vit_large(patch_size=16)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

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

# =========================
# Utils
# =========================
def extract_multiscale(model, img_bgr):
    feats_all = []

    for size in [768, 1024]:
        global IMAGE_SIZE
        IMAGE_SIZE = size

        q = preprocess(img_bgr).unsqueeze(0).to(DEVICE)
        feat = extract_embedding(model, q)
        feats_all.append(feat)

    feat = torch.stack(feats_all).mean(dim=0)
    return feat

def imread_unicode(p):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = IMAGE_SIZE / min(h, w)
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img, (nw, nh))

    # center crop
    y0 = (nh - IMAGE_SIZE) // 2
    x0 = (nw - IMAGE_SIZE) // 2
    img = img[y0:y0+IMAGE_SIZE, x0:x0+IMAGE_SIZE]

    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))

    return torch.from_numpy(img)

def gem_pool(x, p=3.0, eps=1e-6):
    return (x.clamp(min=eps).pow(p).mean(dim=1)).pow(1.0/p)

@torch.no_grad()
def extract_embedding(model, img_tensor):
    feats = model.forward_features(img_tensor)

    if isinstance(feats, dict):
        if "x_norm_patchtokens" in feats and torch.is_tensor(feats["x_norm_patchtokens"]):
            emb = feats["x_norm_clstoken"]
            return F.normalize(emb, dim=1)

        for k in ["x_norm", "tokens", "x", "last_hidden_state", "x_prenorm"]:
            if k in feats and torch.is_tensor(feats[k]) and feats[k].dim() == 3:
                tok = feats[k]                       # [B, T, C]
                patch = tok[:, 1:, :]                # 去 CLS
                emb = gem_pool(patch)
                return F.normalize(emb, dim=1)

        if "x_norm_clstoken" in feats and torch.is_tensor(feats["x_norm_clstoken"]):
            return F.normalize(feats["x_norm_clstoken"], dim=1)

        raise KeyError(list(feats.keys()))

    if torch.is_tensor(feats):
        if feats.dim() == 3:
            patch = feats[:, 1:, :]
            emb = patch.mean(dim=1)
            return F.normalize(emb, dim=1)
        if feats.dim() == 2:
            return F.normalize(feats, dim=1)
        raise ValueError(feats.shape)

    raise TypeError(type(feats))

def load_faiss(index_path, meta_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    index = faiss.read_index(index_path)
    img_paths = np.load(meta_path, allow_pickle=True)
    return index, img_paths

# =========================
# Search
# =========================
def search_one(model, index, img_paths, query_img_path, topk=10):
    img = imread_unicode(query_img_path)
    if img is None:
        raise ValueError(f"Cannot read query image: {query_img_path}")

    q = preprocess(img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    q_feat = extract_multiscale(model, img).cpu().numpy().astype("float32")

    # IndexFlatIP: returns (scores, ids)
    scores, ids = index.search(q_feat, topk)

    results = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx < 0:
            continue
        results.append((rank, float(score), str(img_paths[idx])))
    return results


def _fit_to_box(img_bgr, box_w, box_h):
    """等比缩放并居中贴到固定大小黑底框"""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((box_h, box_w, 3), np.uint8)

    scale = min(box_w / w, box_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((box_h, box_w, 3), np.uint8)
    x0 = (box_w - nw) // 2
    y0 = (box_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def make_retrieval_montage(
    query_path: str,
    results: list,  # [(rank, score, path), ...]
    out_path: str,
    cols: int = 4,
    cell_w: int = 260,
    cell_h: int = 360,
    margin: int = 30,
    header_h: int = 60,
    bg_color=(0, 0, 0),
    font_scale=1.2,
    thickness=3,
):
    """
    生成类似示例图的拼图：
    - 第一个格子放 QUERY
    - 后面按 rank 顺序放 #i score
    """
    # 组装所有要展示的条目：query + topK
    items = [("QUERY", None, query_path)]
    for rank, score, path in results:
        items.append((f"#{rank}", float(score), path))

    n = len(items)
    rows = math.ceil(n / cols)

    W = margin + cols * (cell_w + margin)
    H = margin + rows * (cell_h + header_h + margin)

    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = bg_color

    for idx, (title, score, path) in enumerate(items):
        r = idx // cols
        c = idx % cols

        x = margin + c * (cell_w + margin)
        y = margin + r * (cell_h + header_h + margin)

        # 读图（支持中文路径用 np.fromfile + imdecode）
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((cell_h, cell_w, 3), np.uint8)

        tile = _fit_to_box(img, cell_w, cell_h)
        canvas[y + header_h:y + header_h + cell_h, x:x + cell_w] = tile

        # 写字
        if title == "QUERY":
            text = "QUERY"
        else:
            text = f"{title}  {score:.3f}"

        # 左上角写字（白色）
        cv2.putText(
            canvas,
            text,
            (x, y + int(header_h * 0.75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(out_path, canvas)
    return out_path

def main():
    print("[LOAD] faiss index + meta ...")
    index, img_paths = load_faiss(GLOBAL_INDEX, GLOBAL_META)
    print("[LOAD] index ntotal:", index.ntotal, "meta:", len(img_paths))

    print("[LOAD] model ...")
    model = build_dinov3_vitl16(CKPT, DEVICE)

    print("[SEARCH] query:", QUERY_IMG)
    results = search_one(model, index, img_paths, QUERY_IMG, TOPK)
    # results = search_one(...)
    out = make_retrieval_montage(
        query_path=QUERY_IMG,
        results=results,
        out_path=r"D:\zhanlan\faiss_dinov3\montage.png",
        cols=4,  # 每行4张（跟你截图类似）
        cell_w=260,  # 单格宽
        cell_h=360,  # 单格高（竖图更友好）
        margin=30,
        header_h=60
    )
    print("Saved montage:", out)

    print("\n===== TOPK RESULTS =====")
    for rank, score, path in results:
        print(f"{rank:02d}  score={score:.4f}  {path}")

if __name__ == "__main__":
    main()
