import numpy as np
import os

base_dir = r".\outputs_hybrid_folder_big"

# 加载映射
clip_g_map = np.load(os.path.join(base_dir, "clip_g_vec_to_img.npy"))
dino_g_map = np.load(os.path.join(base_dir, "dinov3_g_vec_to_img.npy"))

print(f"CLIP Global Vectors: {len(clip_g_map)}")
print(f"DINO Global Vectors: {len(dino_g_map)}")

# 检查前 10 个和后 10 个
print("First 10 IDs match:", np.array_equal(clip_g_map[:10], dino_g_map[:10]))
print("Last 10 IDs match:", np.array_equal(clip_g_map[-10:], dino_g_map[-10:]))

# 随机检查
import random
indices = random.sample(range(len(clip_g_map)), 5)
for i in indices:
    print(f"Index {i}: CLIP={clip_g_map[i]}, DINO={dino_g_map[i]}, Match={clip_g_map[i]==dino_g_map[i]}")