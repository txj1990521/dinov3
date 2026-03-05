import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 模型路径 (必须与主建库脚本一致)
MODEL_ID = "models/facebook/dinov3-vitl16-pretrain-lvd1689m"
# MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# 现有的索引文件和路径文件 (必须与主建库脚本生成的文件名一致)
EXISTING_INDEX_PATH = "fabric_index_multi.faiss"
EXISTING_PATHS_FILE = "image_paths_multi.npy"

# 【用户修改区】在这里填入你要新增的文件夹路径
NEW_DATASET_ROOTS = [
    r"D:\zhanlan\自己拍摄的花色图",
]


# ===========================================

def load_model():
    print("⏳ 正在加载模型...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
        print(f"✅ 模型加载成功！运行设备: {device}")
        return processor, model, device
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()


def load_existing_index():
    """加载现有的索引和路径列表"""
    if not os.path.exists(EXISTING_INDEX_PATH) or not os.path.exists(EXISTING_PATHS_FILE):
        print(f"❌ 错误：未找到现有的索引文件 ({EXISTING_INDEX_PATH})。")
        print("💡 提示：请先运行主建库脚本 (build_fabric_index.py) 创建初始库。")
        return None, None

    print(f"📂 正在加载现有索引: {EXISTING_INDEX_PATH} ...")
    index = faiss.read_index(EXISTING_INDEX_PATH)
    paths = np.load(EXISTING_PATHS_FILE, allow_pickle=True)

    # 转换为 set 以便快速查重
    existing_paths_set = set(str(p) for p in paths)

    print(f"✅ 现有库容量: {len(paths)} 张图片")
    return index, paths, existing_paths_set


def scan_new_files(root_folders, existing_paths_set):
    """扫描新文件夹，并过滤掉已存在的图片"""
    new_files = []
    skipped_count = 0

    print("🔍 正在扫描新目录并去重...")

    for root in root_folders:
        if not os.path.exists(root):
            print(f"⚠️ 警告：目录不存在，跳过 -> {root}")
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.abspath(os.path.join(dirpath, filename))

                    # 核心去重逻辑
                    if full_path in existing_paths_set:
                        skipped_count += 1
                        continue

                    new_files.append(full_path)

    print(f"✅ 发现新图片: {len(new_files)} 张")
    if skipped_count > 0:
        print(f"⏭️  跳过已存在的图片: {skipped_count} 张")

    return new_files


def extract_and_merge(file_list, processor, model, device, existing_index):
    if not file_list:
        print("💡 没有新图片需要处理，退出。")
        return existing_index, []

    new_embeddings = []
    new_paths = []

    print("🚀 开始提取新图片特征...")

    with torch.no_grad():
        pbar = tqdm(file_list, desc="提取新特征")
        for full_path in pbar:
            try:
                image = Image.open(full_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model(**inputs)

                feature = outputs.pooler_output.cpu().numpy()
                new_embeddings.append(feature)
                new_paths.append(full_path)

            except Exception as e:
                # 跳过坏图
                pass

    if len(new_embeddings) == 0:
        print("❌ 未能从新图片中提取任何有效特征。")
        return existing_index, []

    # 转换为 numpy 并归一化
    new_embeddings_np = np.vstack(new_embeddings).astype('float32')
    faiss.normalize_L2(new_embeddings_np)

    # 合并到现有索引
    print("🔗 正在合并索引...")
    existing_index.add(new_embeddings_np)

    print(f"✅ 成功添加 {len(new_paths)} 张新图片到索引。")
    return existing_index, new_paths


if __name__ == "__main__":
    # 1. 加载模型
    processor, model, device = load_model()

    # 2. 加载现有索引
    result = load_existing_index()
    if result is None:
        exit()

    current_index, old_paths, existing_set = result

    # 3. 扫描新文件 (自动去重)
    new_files = scan_new_files(NEW_DATASET_ROOTS, existing_set)

    if not new_files:
        print("🎉 操作完成！没有新数据需要添加。")
        exit()

    # 4. 提取特征并合并
    updated_index, added_paths = extract_and_merge(new_files, processor, model, device, current_index)

    if not added_paths:
        exit()

    # 5. 保存更新后的索引和路径
    print("💾 正在保存更新后的数据库...")

    # 保存索引
    faiss.write_index(updated_index, EXISTING_INDEX_PATH)

    # 合并路径列表并保存
    # 注意：old_paths 可能是 numpy 数组，需要转 list 再 extend
    all_paths_list = list(old_paths) + added_paths
    np.save(EXISTING_PATHS_FILE, np.array(all_paths_list, dtype=object))

    print("\n" + "=" * 40)
    print(f"🎉 增量更新完成！")
    print(f"📊 新增图片数: {len(added_paths)}")
    print(f"📈 总库容量: {len(all_paths_list)}")
    print(f"💾 索引文件已更新: {EXISTING_INDEX_PATH}")
    print(f"🗺️ 路径文件已更新: {EXISTING_PATHS_FILE}")
    print("=" * 40)