import os
import json
import math
from typing import List, Dict, Any

import numpy as np
import torch
import faiss

from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from tqdm import tqdm

# =========================
# 配置区：改这里就行
# =========================
CFG = {
    # 模型路径
    "model": r"D:\zhanlanProject\dinov3\models\facebook\dinov3-vitl16-pretrain-lvd1689m",

    # 【注意】后缀是 .jsonl，代码已适配自动识别
    "meta_file": r"D:\zhanlanProject\openai_search\outputs_hybrid_folder_big\images_meta.json",

    # 输出文件
    "index_path": r"D:\data\db\index_m3k.faiss",
    "output_meta_path": r"D:\data\db\img_meta_m3k.json",

    # 索引类型：flat / ivf
    "index_type": "ivf",

    # 批量推理
    "batch_size": 64,

    # IVF 训练用多少张图
    "train_size": 6000,

    # IVF 聚类中心数
    "nlist": 256,

    # 是否归一化
    "normalize": False,

    # 设备
    "device": None,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image_paths_from_meta(meta_path: str) -> List[Dict[str, Any]]:
    """
    从 JSON 或 JSONL 文件中加载图片信息。
    自动检测文件格式：
    - 如果扩展名是 .jsonl 或 .jsonl.gz，按行读取。
    - 否则尝试作为标准 JSON 读取。
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    items = []
    ext = os.path.splitext(meta_path)[1].lower()

    # 判断是否为 JSONL 格式
    is_jsonl = (ext == '.jsonl' or ext == '.jsonlines')

    try:
        if is_jsonl:
            print("Detected JSONL format. Reading line by line...")
            with open(meta_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        items.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
        else:
            # 尝试作为标准 JSON 读取
            print("Detected standard JSON format. Loading whole file...")
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                if "imgs" in data:
                    items = data["imgs"]
                elif "data" in data:
                    items = data["data"]
                else:
                    values = list(data.values())
                    if len(values) == 1 and isinstance(values[0], list):
                        items = values[0]
                    else:
                        raise ValueError("Unsupported JSON structure.")
            else:
                raise ValueError("JSON root must be a list or dict.")

    except Exception as e:
        # 如果标准 JSON 读取失败，且用户没强制指定 .jsonl，可以尝试 fallback 到按行读取（以防扩展名不对但内容是 jsonl）
        if not is_jsonl:
            print(f"Standard JSON load failed ({e}). Trying to read as JSONL fallback...")
            with open(meta_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line: continue
                    try:
                        items.append(json.loads(line))
                    except:
                        continue
            if not items:
                raise e  # 如果 fallback 也没读到数据，抛出原始错误
        else:
            raise e

    # 验证并过滤有效路径
    valid_items = []
    for item in items:
        if not isinstance(item, dict):
            continue

        # 兼容多种路径键名
        path = item.get("abs_path") or item.get("path") or item.get("file_path")

        if path:
            # 检查文件是否存在 (可选，如果文件太多可能慢，但为了稳健性保留)
            if os.path.exists(path):
                if "img_id" not in item:
                    item["img_id"] = len(valid_items)
                valid_items.append(item)
            else:
                # 如果路径不存在，可以选择跳过或保留（取决于你的需求，这里选择跳过并警告）
                # 如果这是网络路径或相对路径问题，可以注释掉 os.path.exists 检查
                print(f"Warning: Path not found on disk: {path}")
                # 如果你确定路径逻辑没问题只是当前环境访问不到，可以取消下面这行的注释来强制加入
                # valid_items.append(item)
        else:
            print(f"Warning: Item missing path field: {item}")

    if not valid_items:
        raise RuntimeError("No valid items with existing paths found in the meta file.")

    # 按 img_id 排序
    valid_items.sort(key=lambda x: x.get("img_id", 0))

    print(f"Successfully loaded {len(valid_items)} items.")
    return valid_items


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
    def infer_batch_features(self, image_inputs: List[str]) -> torch.Tensor:
        images = [load_image(p) for p in image_inputs]
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output


def gen_batch(items: List[Dict], batch_size: int):
    total = len(items)
    batches = math.ceil(total / batch_size)
    for b in range(batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, total)
        batch_items = items[s:e]
        batch_ids = np.array([item["img_id"] for item in batch_items], dtype=np.int64)
        batch_paths = [item["abs_path"] for item in batch_items]
        yield b + 1, batches, batch_paths, batch_ids


def save_meta(output_path: str, original_data: List[Dict]):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)
    print(f"Saved meta to: {output_path}")


def maybe_normalize(x: np.ndarray, do_norm: bool) -> np.ndarray:
    if not do_norm:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_index(cfg: dict):
    model_name_or_path = cfg["model"]
    meta_file = cfg["meta_file"]
    index_path = cfg["index_path"]
    output_meta_path = cfg.get("output_meta_path", index_path.replace(".faiss", "_meta.json"))

    index_type = cfg["index_type"]
    batch_size = int(cfg["batch_size"])
    train_size = int(cfg["train_size"])
    nlist = int(cfg["nlist"])
    normalize = bool(cfg["normalize"])
    device = cfg.get("device", None)

    # 1. 加载元数据
    print(f"Loading meta from: {meta_file}")
    img_items = load_image_paths_from_meta(meta_file)

    if not img_items:
        raise RuntimeError("No valid images found in the meta file.")

    print(f"Found {len(img_items)} valid images.")
    save_meta(output_meta_path, img_items)

    extractor = DINOv3FeatureExtractor(model_name_or_path, device=device)

    # 2. 准备训练数据 (仅 IVF)
    train_x = None
    if index_type == "ivf":
        train_items = img_items[: min(train_size, len(img_items))]
        print(f"Training IVF with {len(train_items)} samples...")

        train_vecs = []
        for b_idx, b_total, b_paths, _ in tqdm(gen_batch(train_items, batch_size),
                                               total=math.ceil(len(train_items) / batch_size), desc="Training IVF"):
            feats = extractor.infer_batch_features(b_paths).float().cpu().numpy().astype(np.float32)
            feats = maybe_normalize(feats, normalize)
            train_vecs.append(feats)

        train_x = np.concatenate(train_vecs, axis=0).astype(np.float32)
        del train_vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. 获取特征维度
    if train_x is not None:
        d = train_x.shape[1]
    else:
        first_path = [img_items[0]["abs_path"]]
        one = extractor.infer_batch_features(first_path).float().cpu().numpy().astype(np.float32)
        one = maybe_normalize(one, normalize)
        d = one.shape[1]

    print("Feature dimension:", d)

    # 4. 创建 Index
    if index_type == "flat":
        index = faiss.IndexFlatL2(d)
        print("Index Type: IndexFlatL2")
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(train_x)
        print(f"Index Type: IndexIVFFlat (nlist={nlist})")
    else:
        raise ValueError("index_type must be 'flat' or 'ivf'")

    # 5. 全量入库
    total_batches = math.ceil(len(img_items) / batch_size)
    print("Start indexing all images...")

    for b_idx, b_total, b_paths, b_ids in tqdm(gen_batch(img_items, batch_size),
                                               total=total_batches, desc="Indexing"):
        feats = extractor.infer_batch_features(b_paths).float().cpu().numpy().astype(np.float32)
        feats = maybe_normalize(feats, normalize)
        index.add_with_ids(feats, b_ids)

    # 6. 保存
    out_dir = os.path.dirname(index_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    faiss.write_index(index, index_path)
    print("Done!")
    print(f"Saved index to: {index_path}")
    print(f"Saved meta to : {output_meta_path}")


def main():
    build_index(CFG)


if __name__ == "__main__":
    main()