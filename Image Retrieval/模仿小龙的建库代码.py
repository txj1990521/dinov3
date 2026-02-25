# build_index.py
import os
import json
import math
from typing import List

import numpy as np
import torch
import faiss

from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


# =========================
# 配置区：改这里就行
# =========================
CFG = {
    # 模型（可以是 modelscope/hf 的模型名，也可以是本地目录）
    "model": r"D:\zhanlanProject\dinov3\models\facebook\dinov3-vitl16-pretrain-lvd1689m",

    # 图库根目录（会递归扫描所有图片）
    "img_root": r"D:\zhanlan\new_data_noCrop",

    # 输出文件
    "index_path": r"D:\data\db\index_m3k.faiss",
    "meta_path":  r"D:\data\db\img_meta_m3k.json",

    # 索引类型：flat（最准最慢） / ivf（更快，适合大库）
    "index_type": "ivf",   # "flat" or "ivf"

    # 批量推理（显存不够就调小）
    "batch_size": 64,

    # IVF 训练用多少张图（越多召回越好，但建库更慢）
    "train_size": 20000,

    # IVF 聚类中心数（越大越准但更慢；一般 100~4096 之间）
    "nlist": 256,

    # IVF 训练后入库前是否归一化特征（如果你未来改用余弦相似度会有用）
    "normalize": False,

    # 强制设备：None=自动；"cpu" 强制 CPU；"cuda" 强制 GPU（有CUDA时）
    "device": None,  # None / "cpu" / "cuda"
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: str) -> List[str]:
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dp, fn))
    paths.sort()
    return paths


class DINOv3FeatureExtractor:
    def __init__(self, pretrained_model_name: str, device: str | None = None):
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        # device_map="auto" 会自己挑 GPU/CPU；但你也可以强制
        if device == "cpu":
            self.model = AutoModel.from_pretrained(pretrained_model_name).to("cpu")
        elif device == "cuda":
            self.model = AutoModel.from_pretrained(pretrained_model_name).to("cuda")
        else:
            self.model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")

        self.model.eval()
        print("model device:", self.model.device)

    @torch.inference_mode()
    def infer_batch_features(self, image_inputs: List[str]) -> torch.Tensor:
        images = [load_image(p) for p in image_inputs]
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output  # [B, D]


def gen_batch(items: List[str], batch_size: int):
    total = len(items)
    batches = math.ceil(total / batch_size)
    for b in range(batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, total)
        yield b + 1, batches, items[s:e], np.arange(s, e, dtype=np.int64)


def save_meta(meta_path: str, img_paths: List[str]):
    data = {"imgs": [[str(i), p] for i, p in enumerate(img_paths)]}
    out_dir = os.path.dirname(meta_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def maybe_normalize(x: np.ndarray, do_norm: bool) -> np.ndarray:
    if not do_norm:
        return x
    # L2 normalize
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_index(cfg: dict):
    model_name_or_path = cfg["model"]
    img_root = cfg["img_root"]
    index_path = cfg["index_path"]
    meta_path = cfg["meta_path"]
    index_type = cfg["index_type"]
    batch_size = int(cfg["batch_size"])
    train_size = int(cfg["train_size"])
    nlist = int(cfg["nlist"])
    normalize = bool(cfg["normalize"])
    device = cfg.get("device", None)

    img_paths = list_images(img_root)
    if not img_paths:
        raise RuntimeError(f"No images found under: {img_root}")

    print(f"Found {len(img_paths)} images.")
    save_meta(meta_path, img_paths)

    extractor = DINOv3FeatureExtractor(model_name_or_path, device=device)

    # 1) 训练向量（仅 IVF 需要）
    train_x = None
    if index_type == "ivf":
        train_paths = img_paths[: min(train_size, len(img_paths))]
        print(f"Training sample size: {len(train_paths)}")

        train_vecs = []
        for b_idx, b_total, b_imgs, _ in gen_batch(train_paths, batch_size):
            feats = extractor.infer_batch_features(b_imgs).float().cpu().numpy().astype(np.float32)
            feats = maybe_normalize(feats, normalize)
            train_vecs.append(feats)
            print(f"train batch {b_idx}/{b_total}: {feats.shape}")

        train_x = np.concatenate(train_vecs, axis=0).astype(np.float32)

    # 2) 得到特征维度 d（flat 也要）
    if train_x is not None:
        d = train_x.shape[1]
    else:
        # flat 模式：用第一张图拿到维度
        one = extractor.infer_batch_features([img_paths[0]]).float().cpu().numpy().astype(np.float32)
        one = maybe_normalize(one, normalize)
        d = one.shape[1]

    print("feature dim:", d)

    # 3) 创建 index
    if index_type == "flat":
        # L2 距离
        index = faiss.IndexFlatL2(d)
        print("Index: IndexFlatL2")
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(train_x)
        print(f"Index: IndexIVFFlat (nlist={nlist})")
    else:
        raise ValueError("index_type must be 'flat' or 'ivf'")

    # 4) 全量入库（统一 add_with_ids）
    for b_idx, b_total, b_imgs, b_ids in gen_batch(img_paths, batch_size):
        feats = extractor.infer_batch_features(b_imgs).float().cpu().numpy().astype(np.float32)
        feats = maybe_normalize(feats, normalize)
        index.add_with_ids(feats, b_ids)
        print(f"add batch {b_idx}/{b_total}: {feats.shape}, ids {b_ids[0]}~{b_ids[-1]}")

    out_dir = os.path.dirname(index_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    faiss.write_index(index, index_path)
    print("Saved index to:", index_path)
    print("Saved meta to :", meta_path)


def main():
    build_index(CFG)


if __name__ == "__main__":
    main()