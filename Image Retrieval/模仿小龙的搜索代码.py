# search_index.py
import os
import json
from typing import List, Union

import numpy as np
import torch
import faiss
from PIL import Image

from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


# =========================
# 配置区：改这里就行
# =========================
CFG = {
    # 必须与建库时一致（同一个模型/同一套特征）
    "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",

    # 建库输出的文件
    "index_path": r"D:\data\db\index_m3k.faiss",
    "meta_path":  r"D:\data\db\img_meta_m3k.json",

    # 查询图片（单张）
    "query_img": r"D:\data\query\q1.jpg",

    # 返回 topk
    "topk": 5,

    # IVF 召回关键参数：只对 IVF 索引生效，flat 忽略
    "nprobe": 20,

    # 是否对查询特征做 L2 归一化（必须和建库时 normalize 一致）
    "normalize": False,

    # 强制设备：None=自动；"cpu" 强制 CPU；"cuda" 强制 GPU（有CUDA时）
    "device": None,  # None / "cpu" / "cuda"
}


class DINOv3FeatureExtractor:
    def __init__(self, pretrained_model_name: str, device: str | None = None):
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        if device == "cpu":
            self.model = AutoModel.from_pretrained(pretrained_model_name).to("cpu")
        elif device == "cuda":
            self.model = AutoModel.from_pretrained(pretrained_model_name).to("cuda")
        else:
            self.model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")

        self.model.eval()
        print("model device:", self.model.device)

    @torch.inference_mode()
    def infer_batch_features(self, image_inputs: List[Union[str, Image.Image]]) -> torch.Tensor:
        images = []
        for x in image_inputs:
            images.append(load_image(x) if isinstance(x, str) else x)
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output  # [B, D]


def load_meta(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data["imgs"] = [["0","/path"], ...]
    return data["imgs"]


def maybe_normalize(x: np.ndarray, do_norm: bool) -> np.ndarray:
    if not do_norm:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def search(cfg: dict):
    model = cfg["model"]
    index_path = cfg["index_path"]
    meta_path = cfg["meta_path"]
    query_img = cfg["query_img"]
    topk = int(cfg["topk"])
    nprobe = int(cfg["nprobe"])
    normalize = bool(cfg["normalize"])
    device = cfg.get("device", None)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"index not found: {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta not found: {meta_path}")
    if not os.path.exists(query_img):
        raise FileNotFoundError(f"query image not found: {query_img}")

    extractor = DINOv3FeatureExtractor(model, device=device)
    index = faiss.read_index(index_path)

    # IVF 索引有 nprobe；flat 没有
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
        print("index type: IVF-like, nprobe =", index.nprobe)
    else:
        print("index type: Flat-like")

    meta = load_meta(meta_path)

    q_feat = extractor.infer_batch_features([query_img]).float().cpu().numpy().astype(np.float32)
    q_feat = maybe_normalize(q_feat, normalize)

    distances, ids = index.search(q_feat, topk)

    print("\n====================")
    print("query:", query_img)
    print("====================")
    for rank, (dist, idx) in enumerate(zip(distances[0], ids[0]), start=1):
        if idx < 0:
            continue
        # idx 对应 meta 的行号（因为建库时 add_with_ids 用的是 0..N-1）
        path = meta[idx][1] if idx < len(meta) else "<out_of_range>"
        print(f"top{rank}  id={idx}  dist={dist:.6f}  path={path}")


def main():
    search(CFG)


if __name__ == "__main__":
    main()