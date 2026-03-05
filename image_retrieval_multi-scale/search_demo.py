import os
import json
import faiss
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

# 和提特征时一致
REPO_DIR = r"YOUR_LOCAL_PATH_TO\dinov3"
WEIGHTS  = r"YOUR_WEIGHTS_PATH_OR_URL"
MODEL_NAME = "dinov3_vitb16"
SCALES = (384, 512)
USE_LAST_N_LAYERS = 1
FUSE_ALPHA = 0.5
GEM_P = 3.0

OUT_DIR = r"D:\dinov3_retrieval_out"
INDEX_FILE = os.path.join(OUT_DIR, "faiss_flatip.index")
PATHS_FILE = os.path.join(OUT_DIR, "paths.jsonl")

def make_transform(size: int):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def gem_pool(feat_map: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    x = feat_map.clamp(min=eps).pow(p)
    x = x.mean(dim=(-1, -2)).pow(1.0 / p)
    return x

@torch.inference_mode()
def extract_one(model, img: Image.Image, device: str):
    embs = []
    for size in SCALES:
        x = make_transform(size)(img.convert("RGB")).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            outs = model.get_intermediate_layers(
                x, n=USE_LAST_N_LAYERS, reshape=True, return_class_token=True, norm=True
            )
        patch_maps, clss = [], []
        for (pm, cls) in outs:
            patch_maps.append(pm)
            clss.append(cls)
        patch_map = torch.stack(patch_maps, dim=0).mean(dim=0)
        cls = torch.stack(clss, dim=0).mean(dim=0)

        patch_vec = gem_pool(patch_map, p=GEM_P)
        emb = FUSE_ALPHA * cls + (1.0 - FUSE_ALPHA) * patch_vec
        emb = F.normalize(emb, p=2, dim=-1)
        embs.append(emb)

    emb = torch.stack(embs, dim=0).mean(dim=0)
    emb = F.normalize(emb, p=2, dim=-1).squeeze(0).float().cpu().numpy().astype("float32")
    return emb

def load_paths():
    paths = []
    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            paths.append(obj["path"])
    return paths

def main(query_img_path: str, topk: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load(REPO_DIR, MODEL_NAME, source="local", weights=WEIGHTS).to(device).eval()

    index = faiss.read_index(INDEX_FILE)
    paths = load_paths()

    qimg = Image.open(query_img_path).convert("RGB")
    q = extract_one(model, qimg, device=device)
    faiss.normalize_L2(q.reshape(1, -1))

    scores, ids = index.search(q.reshape(1, -1), topk)
    scores, ids = scores[0], ids[0]

    print("TopK results:")
    for rank, (idx, s) in enumerate(zip(ids, scores), start=1):
        print(f"{rank:02d}  score={float(s):.4f}  path={paths[idx]}")

if __name__ == "__main__":
    # 改成你的手机拍的查询图路径
    main(r"D:\your_query.jpg", topk=10)