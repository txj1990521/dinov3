import math
import os.path
import json
import sys

import torch
from modelscope import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
import torchvision.datasets as datasets
from PIL import Image
from typing import List, Union
import faiss
import numpy as np


class DINOv3FeatureExtractor:
    def __init__(self, pretrained_model_name: str):
        """
        初始化DINOv3模型和处理器。

        参数:
            pretrained_model_name (str): 预训练模型的路径或名称。
        """
        # 加载预处理工具和模型
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
        print("self.model.device:", self.model.device)
        self.model.eval()  # 设置为评估模式

    def infer_batch_features(self, image_inputs: List[Union[str, Image.Image]]) -> torch.Tensor:
        """
        对给定的一批图像进行推理，返回提取的特征。

        参数:
            image_inputs (List[Union[str, Image.Image]]): 包含图像的本地路径、URL 或已加载的 PIL 图像列表。

        返回:
            torch.Tensor: 模型输出的 pooled 特征，形状为 [batch_size, hidden_size]。
        """
        images = []
        for image_input in image_inputs:
            if isinstance(image_input, str):
                image = load_image(image_input)
            else:
                image = image_input  # 假设已经是 PIL.Image.Image
            images.append(image)

        # 预处理所有图像并移至模型设备
        inputs = self.processor(images=images, return_tensors="pt").to(self.model.device)

        # 推理
        with torch.inference_mode():
            outputs = self.model(**inputs)

        # 返回 pooled 输出
        return outputs.pooler_output


def init_index_flat(xb: np):
    _, d = xb.shape
    index = faiss.IndexFlatL2(d)  # 使用L2距离
    index.add(xb)  # 将向量添加到索引中
    return index


def init_index_IVF(xb: np):
    """
    IVF索引
    :param xb:
    :return:
    """
    _, d = xb.shape
    nlist = 100  # 量化中心的数量
    quantizer = faiss.IndexFlatL2(d)  # 量化器
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(xb[:10000])  # 使用部分数据训练量化器
    index.add(xb)  # 将向量添加到索引中
    return index


def load_imgs(imgs_path):
    imgs = datasets.ImageFolder(root=imgs_path, transform=None)
    # imgs = [img_file for img_file, _ in imgs.imgs[:20]]
    imgs = [img_file for img_file, _ in imgs.imgs]
    return imgs


def export_imgs_meta(imgs_meta_file,imgs):
    with open(imgs_meta_file,"w",encoding="utf-8") as wf:
        wf.write("{\"imgs\":[")
        wf.write(f"\n[\"0\",\"{imgs[0]}\"]")
        for idx, img_file in enumerate(imgs[1:]):
            wf.write(f",\n[\"{idx+1}\",\"{img_file}\"]")
        wf.write("\n]}")


def read_json_file(json_file):
    json_obj=None
    if os.path.exists(json_file) and os.path.isfile(json_file):
        with open(json_file,"r",encoding="utf-8") as rf:
            json_obj = json.load(rf)
    return json_obj


def load_imgs_meta(imgs_meta_file):
    json_obj=None
    if os.path.exists(imgs_meta_file) and os.path.isfile(imgs_meta_file):
        with open(imgs_meta_file, "r", encoding="utf-8") as rf:
            json_obj = json.load(rf)
    return json_obj


def load_dinov3(model_path):
    extractor = DINOv3FeatureExtractor(model_path)
    return extractor


def save2index(index_file,index):
    faiss.write_index(index,index_file)


def load_index(index_file):
    index = faiss.read_index(index_file)
    return index


def gen_batch(imgs, batch_size):
    total_img_num = len(imgs)
    batch_num = math.ceil(total_img_num * 1.0 / batch_size)
    for b_idx in range(batch_num):
        bg_idx, ed_idx = b_idx*batch_size, b_idx*batch_size + batch_size
        if total_img_num <= ed_idx:
            ed_idx = total_img_num
        batch_imgs = imgs[bg_idx:ed_idx]
        # print(f"b_idx:{b_idx}",batch_imgs)
        idxs_np = np.arange(bg_idx, ed_idx)
        # print(idxs_np)
        yield batch_num, b_idx+1, batch_imgs,idxs_np


def index_db_init(imgs_path, imgs_meta_file, index_file, model_path, batch_size):
    print("index_db_init")
    imgs = load_imgs(imgs_path)
    export_imgs_meta(imgs_meta_file, imgs)
    extractor = load_dinov3(model_path)
    total_img_num = len(imgs)
    print(f"total_img_num:{total_img_num}")
    if len(imgs) < batch_size:
        print(f"all in one batch")
        features = extractor.infer_batch_features(imgs)
        fts_np = features.cpu().numpy()
        print("fts_np.shape:\n",fts_np.shape)
        # index = init_index_flat(fts_np)
        index = init_index_IVF(fts_np)
        save2index(index_file, index)
    else:
        print(f"total_img_num:{total_img_num}")
        index = None
        for total_batch, cu_batch_idx, batch_imgs, batch_idxs_np in gen_batch(imgs, batch_size):
            print(f"total_batch:{total_batch},cu_batch_idx:{cu_batch_idx},{batch_idxs_np[0]}~{batch_idxs_np[-1]}")
            features = extractor.infer_batch_features(batch_imgs)
            fts_np = features.cpu().numpy()
            print("fts_np.shape:\n", fts_np.shape)
            if index is None:
                # index = init_index_flat(fts_np)
                index = init_index_IVF(fts_np)
            else:
                index.add_with_ids(fts_np, batch_idxs_np)
        save2index(index_file, index)


def query_sim(model_path, imgs_meta_file, index_file, q_img_file):
    print(f"query_sim,q_img_file:{q_img_file}")
    extractor = load_dinov3(model_path)
    index = load_index(index_file)
    features = extractor.infer_batch_features([q_img_file])
    fts_np = features.cpu().numpy()
    print("fts_np.shape:\n", fts_np.shape)
    distances, labels = index.search(fts_np, 5)  # 进行搜索
    # distances, labels = index.search(fts_np[0], 2)  # 进行搜索
    # distances, labels = index.search(features, 2)  # 进行搜索
    print("distances:", distances)  # 输出相似度（距离）
    print("labels:", labels)  # 输出相似向量的索引
    imgs_meta = load_imgs_meta(imgs_meta_file)
    # print("imgs_meta:\n",imgs_meta)
    imgs_meta = imgs_meta["imgs"]
    print(f"q_img_file:{q_img_file}")
    for idx, label in enumerate(labels[0]):
        print(f"idx,label:{idx},{label}")
        sim_v = distances[0][idx]
        sim_img_file = imgs_meta[label][1]
        print(f"top_{idx},{sim_v},{sim_img_file}")


def simple_test(index_file):
    fts_np = np.array([
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
        [1.0,1.0,1.0,1.0,1.0,1.0],
    ], dtype=np.float32)
    # fts_np = fts_np[:,:5]
    print("fts_np:\n",fts_np)
    index = init_index_flat(fts_np)
    # index = init_index_IVF(fts_np)
    save2index(index_file, index)

    distances, labels = index.search(fts_np, 2)
    print(labels)


def test_query(q_imgs_path, imgs_meta_file, index_file, model_path):
    # extractor = load_dinov3(model_path)
    imgs = load_imgs(q_imgs_path)
    q_img_file = imgs[0]
    query_sim(model_path, imgs_meta_file, index_file, q_img_file)


def test():
    imgs = ["img0","img1","img2","img3","img4","img5","img6","img7","img8","img9","img10"]
    batch_size = 3
    for total_batch, cu_batch_idx, batch_imgs, batch_idxs_np in gen_batch(imgs,batch_size):
        print(f"total_batch:{total_batch},cu_batch_idx:{cu_batch_idx},{batch_idxs_np[0]}~{batch_idxs_np[-1]}")


if __name__ == "__main__":
    # 每批次处理图片数量
    batch_size = 200
    # test()
    # sys.exit(1)
    task_key = "m3k_v2_by_vitl16"
    imgs_path = "/opt/data/imgs/new_data"
    # imgs_path = "/Users/xyl/zhanlan/data/pics_search/new_data"

    # model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    model_path = f"/opt/apps/resources/model/{model_name}"
    # model_path = f"/Users/xyl/IdeaProjects/zhanlan/ai_server_test/resources/models/{model_name}"
    db_root_path = f"/opt/apps/resources/model/db"

    imgs_meta_file = os.path.join(db_root_path, f"img_meta_{task_key}.json")
    index_file = os.path.join(db_root_path, f"index_{task_key}.db")

    # simple_test(index_file)

    # index_db_init(imgs_path, imgs_meta_file, index_file, model_path, batch_size)
    #
    q_imgs_path = "/opt/data/imgs"
    # q_imgs_path = "/Users/xyl/zhanlan/data/pics_search/new_data"
    test_query(q_imgs_path, imgs_meta_file, index_file,model_path)

