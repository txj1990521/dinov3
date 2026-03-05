import os
import faiss
import numpy as np

OUT_DIR = r"D:\dinov3_retrieval_out_stream"
EMB_FILE = os.path.join(OUT_DIR, "embeddings.npy")
VALID_FILE = os.path.join(OUT_DIR, "valid_count.txt")
INDEX_FILE = os.path.join(OUT_DIR, "faiss_flatip.index")

def main():
    with open(VALID_FILE, "r", encoding="utf-8") as f:
        valid = int(f.read().strip())

    embs = np.load(EMB_FILE, mmap_mode="r")  # memmap 方式读
    embs = np.asarray(embs[:valid], dtype="float32")  # 只取有效行（拷贝到连续内存更适合 faiss）
    N, D = embs.shape
    print(f"[INFO] Using embeddings: {N} x {D}")

    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(D)
    index.add(embs)
    faiss.write_index(index, INDEX_FILE)
    print(f"[INFO] Saved index: {INDEX_FILE}")

if __name__ == "__main__":
    main()