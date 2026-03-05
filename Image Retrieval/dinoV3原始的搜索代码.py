import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np
import shutil

# ================= 配置区域 =================
# 必须与建库代码中的模型 ID 保持一致
MODEL_ID = "models/facebook/dinov3-vitl16-pretrain-lvd1689m"
# MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m" # 如果本地没有，取消此行注释并联网

# 建库时生成的文件路径
INDEX_PATH = "fabric_index_multi.faiss"
PATHS_FILE = "image_paths_multi.npy"

# 搜索结果展示配置
TOP_K = 10  # 显示前多少个相似结果
SAVE_RESULTS_DIR = "search_results"  # 搜索结果图片复制到的文件夹


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


def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(PATHS_FILE):
        print(f"❌ 错误：找不到索引文件 '{INDEX_PATH}' 或 '{PATHS_FILE}'。")
        print("💡 提示：请先运行建库脚本 (build_fabric_index)。")
        exit()

    print("📂 正在加载索引数据库...")
    index = faiss.read_index(INDEX_PATH)
    image_paths = np.load(PATHS_FILE, allow_pickle=True)
    print(f"✅ 索引加载完成，库中共有 {len(image_paths)} 张图片。")
    return index, image_paths


def get_query_embedding(query_image_path, processor, model, device):
    """提取查询图片的特征向量"""
    if not os.path.exists(query_image_path):
        raise FileNotFoundError(f"查询图片不存在: {query_image_path}")

    try:
        image = Image.open(query_image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            feature = outputs.pooler_output.cpu().numpy()

        # 归一化 (必须与建库时一致)
        faiss.normalize_L2(feature)
        return feature
    except Exception as e:
        raise Exception(f"处理查询图片失败: {e}")


def search_and_display(query_image_path, index, all_paths, processor, model, device, top_k=10):
    print(f"\n🔍 正在搜索与 '{os.path.basename(query_image_path)}' 相似的布料...\n")

    # 1. 提取特征
    try:
        query_vec = get_query_embedding(query_image_path, processor, model, device)
    except Exception as e:
        print(e)
        return

    # 2. FAISS 搜索
    # IndexFlatIP 返回的是内积，对于归一化向量即为余弦相似度
    distances, indices = index.search(query_vec, top_k)

    # 3. 准备结果展示目录
    if os.path.exists(SAVE_RESULTS_DIR):
        shutil.rmtree(SAVE_RESULTS_DIR)
    os.makedirs(SAVE_RESULTS_DIR)

    # 复制查询图到结果目录
    shutil.copy(query_image_path, os.path.join(SAVE_RESULTS_DIR, f"00_QUERY_{os.path.basename(query_image_path)}"))

    print(f"{'排名':<5} | {'相似度':<10} | {'来源文件':<60}")
    print("-" * 85)

    results = []

    for i, idx in enumerate(indices[0]):
        score = distances[0][i]
        img_path = all_paths[idx]

        # 简单的相似度分数转换 (可选，FAISS IP 范围 -1 到 1，越接近 1 越相似)
        # 对于 DINOv3，通常 > 0.8 就非常相似了

        # 提取相对路径或文件名用于显示
        display_name = os.path.basename(img_path)
        if len(img_path) > 55:
            display_name = "..." + img_path[-52:]

        print(f"{i + 1:<5} | {score:.4f}     | {display_name}")

        # 复制结果图片到结果目录，方便查看
        try:
            # 重命名目标文件，带上排名和分数
            file_name = os.path.basename(img_path)
            ext = os.path.splitext(file_name)[1]
            new_filename = f"{i + 1:02d}_score{score:.2f}_{file_name}"

            # 处理文件名过长或非法字符的问题 (Windows 限制)
            safe_filename = "".join([c for c in new_filename if c.isalpha() or c.isdigit() or c in '._- ']).rstrip()
            if len(safe_filename) > 200:
                safe_filename = safe_filename[:200] + ext

            dest_path = os.path.join(SAVE_RESULTS_DIR, safe_filename)
            shutil.copy(img_path, dest_path)
            results.append({"rank": i + 1, "score": score, "path": img_path, "saved_as": safe_filename})
        except Exception as e:
            print(f"   ⚠️ 无法复制图片 {img_path}: {e}")

    print("-" * 85)
    print(f"✅ 搜索完成！结果已保存至文件夹：./{SAVE_RESULTS_DIR}/")
    print("💡 提示：相似度分数越接近 1.0 表示越相似。")

    return results


if __name__ == "__main__":
    # 1. 初始化
    processor, model, device = load_model()
    index, all_paths = load_index()

    # 2. 获取用户输入
    print("\n请输入要查询的布料图片路径:")
    print("(例如: D:\\test\\my_fabric.jpg 或 ./query.jpg)")
    query_path = input("> ").strip().strip('"')  # 去除可能的引号

    if not query_path:
        print("❌ 未输入路径，退出。")
        exit()

    if not os.path.exists(query_path):
        print(f"❌ 文件不存在: {query_path}")
        exit()

    # 3. 执行搜索
    search_and_display(
        query_path,
        index,
        all_paths,
        processor,
        model,
        device,
        top_k=TOP_K
    )