import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 模型路径 (如果是本地路径请确保存在，否则使用云端 ID)
MODEL_ID = "models/facebook/dinov3-vitl16-pretrain-lvd1689m"
# MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m" # 如果本地没有，取消此行注释并联网

# 2. 多个数据根目录列表 (注意：不要加最后的逗号)
DATASET_ROOTS = [
    r"D:\zhanlan\poptnc.com",
    r"D:\zhanlan\new_data",
    r"D:\印花",
]

# 3. 输出文件名称
OUTPUT_INDEX_PATH = "fabric_index_multi.faiss"
OUTPUT_PATHS_FILE = "image_paths_multi.npy"


# ===========================================

def load_model():
    print("正在加载模型...")
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        # 自动检测是否有 GPU，没有则用 CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
        print(f"模型加载成功！运行设备: {device}")
        return processor, model, device
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("提示：如果是本地路径错误，请检查 'models/facebook...' 目录是否存在，或改用云端 ID。")
        exit()


def load_and_process_image(image_path, processor, model, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        return inputs
    except Exception as e:
        raise Exception(f"无法读取图片 {image_path}: {e}")


def scan_files(root_folders):
    """扫描所有指定文件夹，返回文件列表"""
    all_files = []
    print("🔍 正在扫描所有目录...")

    for root in root_folders:
        if not os.path.exists(root):
            print(f"⚠️ 警告：目录不存在，跳过 -> {root}")
            continue

        # 递归遍历
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(dirpath, filename)
                    all_files.append(full_path)

    print(f"✅ 共发现 {len(all_files)} 张有效图片。")
    return all_files


def build_fabric_index(file_list, processor, model, device, index_path, paths_file):
    if not file_list:
        print("❌ 没有文件可处理，退出。")
        return

    embeddings = []
    valid_paths = []

    print("🚀 开始提取特征 (这可能需要一些时间)...")

    with torch.no_grad():
        pbar = tqdm(file_list, desc="处理进度")
        for full_path in pbar:
            try:
                inputs = load_and_process_image(full_path, processor, model, device)
                outputs = model(**inputs)

                # 获取全局特征
                feature = outputs.pooler_output.cpu().numpy()

                embeddings.append(feature)
                valid_paths.append(os.path.abspath(full_path))

                # 更新进度条描述
                pbar.set_postfix({"当前": os.path.basename(full_path)[:20]})

            except Exception as e:
                # 遇到坏图跳过，不打断整个流程
                # print(f"\n跳过: {e}")
                pass

    if len(embeddings) == 0:
        print("❌ 未能成功提取任何特征。")
        return

    # 转换为 numpy
    embeddings = np.vstack(embeddings).astype('float32')

    # 构建 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    # 归一化并添加
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 保存
    faiss.write_index(index, index_path)
    np.save(paths_file, valid_paths)

    print("\n" + "=" * 40)
    print(f"🎉 索引构建完成！")
    print(f"📂 处理目录数: {len(DATASET_ROOTS)}")
    print(f"📊 成功收录: {len(valid_paths)} 张图片")
    print(f"💾 索引文件: {index_path}")
    print(f"🗺️ 路径文件: {paths_file}")
    print("=" * 40)


if __name__ == "__main__":
    # 1. 加载模型
    processor, model, device = load_model()

    # 2. 扫描所有目录获取文件列表
    all_image_files = scan_files(DATASET_ROOTS)

    # 3. 构建索引
    if all_image_files:
        build_fabric_index(
            all_image_files,
            processor,
            model,
            device,
            OUTPUT_INDEX_PATH,
            OUTPUT_PATHS_FILE
        )
    else:
        print("💡 提示：请检查你的 D 盘路径是否正确，或者文件夹内是否有图片。")