import sys
import os
import torch
import subprocess
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QScrollArea,
                             QFileDialog, QMessageBox, QProgressBar, QGroupBox,
                             QGridLayout, QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette
from PIL import Image, ImageOps, ImageDraw  # ===== [YOLO ADD]
# ================= 引入 YOLO 分割模块 =================  # ===== [YOLO ADD]
try:
    from yolo_seg.seg_preprocess import SegPreprocessor, SegConfig
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠️ 警告：未找到 yolo_seg 模块，YOLO 功能将不可用。")
    YOLO_AVAILABLE = False
# =====================================================  # ===== [YOLO ADD]
# ================= 配置区域 =================
MODEL_ID = "models/facebook/dinov3-vitl16-pretrain-lvd1689m"
# MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m" # 本地没有则取消注释

INDEX_PATH = "fabric_index_multi.faiss"
PATHS_FILE = "image_paths_multi.npy"
TOP_K = 12  # 界面显示多少个结果

# ===== [YOLO ADD] ROI 配置 =====
USE_YOLO_ROI_DEFAULT = True
YOLO_MODEL_PATH = r"yolo_seg/best.pt"
YOLO_DEVICE = 0
YOLO_SCORE_THR = 0.5
YOLO_MIN_AREA_FRAC = 0.03
YOLO_PAD = 16
YOLO_DEBUG_OVERLAY = True
# ===========================================
# ===== [YOLO ADD] 工具函数 =====
def mask_to_bbox(mask_u8: np.ndarray):
    if mask_u8 is None:
        return None
    ys, xs = np.where(mask_u8 > 0)
    if ys.size < 20:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2

def clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(1, min(x2, W))
    y2 = max(1, min(y2, H))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return x1, y1, x2, y2

def draw_bbox_overlay(pil_img: Image.Image, bbox, color=(255, 0, 0), width=4) -> Image.Image:
    if not bbox:
        return pil_img
    im = pil_img.copy()
    draw = ImageDraw.Draw(im)
    x1, y1, x2, y2 = bbox
    for _ in range(width):
        draw.rectangle([x1, y1, x2, y2], outline=color)
        x1 += 1; y1 += 1; x2 -= 1; y2 -= 1
    return im

def apply_yolo_roi(pil_img: Image.Image, segger, use_yolo: bool):
    """
    返回 (roi_img, info, debug_img)
    - roi_img: 输入模型的图（可能是裁剪后的 ROI 或原图）
    - info: used_yolo / error / roi_bbox 等
    - debug_img: 原图上画了 bbox 的 debug 图
    """
    W, H = pil_img.size
    info = {"used_yolo": False, "roi_bbox": None, "mask_area_frac": 0.0, "error": None}

    # 默认 debug 就是原图
    debug_img = pil_img

    if (not YOLO_AVAILABLE) or (not use_yolo) or (segger is None):
        return pil_img, info, debug_img

    try:
        seg_cfg = SegConfig(
            score_thr=float(YOLO_SCORE_THR),
            min_area_frac=float(YOLO_MIN_AREA_FRAC),
            bg_mode="mean",
            use_classes=None,
            merge_all=True,
            device=YOLO_DEVICE,
            debug_dir=None
        )

        _clean, mask_u8 = segger.process_pil(pil_img, seg_cfg)
        bbox = mask_to_bbox(mask_u8)
        if bbox is None:
            info["error"] = "no_mask_found"
            return pil_img, info, debug_img

        x1, y1, x2, y2 = bbox
        area = float((mask_u8 > 0).sum())
        frac = area / float(W * H + 1e-9)
        info["mask_area_frac"] = frac

        if frac < float(YOLO_MIN_AREA_FRAC):
            info["error"] = f"area_too_small ({frac:.2f})"
            return pil_img, info, debug_img

        pad = int(YOLO_PAD)
        x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
        bb = clamp_bbox(x1, y1, x2, y2, W, H)
        if bb is None:
            info["error"] = "invalid_bbox_after_pad"
            return pil_img, info, debug_img

        x1, y1, x2, y2 = bb
        roi = pil_img.crop((x1, y1, x2, y2))

        info.update({
            "used_yolo": True,
            "roi_bbox": [x1, y1, x2, y2],
            "original_size": (W, H),
            "roi_size": (x2 - x1, y2 - y1)
        })

        if YOLO_DEBUG_OVERLAY:
            debug_img = draw_bbox_overlay(pil_img, info["roi_bbox"])

        return roi, info, debug_img

    except Exception as e:
        info["error"] = f"yolo_exception: {repr(e)}"
        return pil_img, info, debug_img
# ==============================
class SearchWorker(QThread):
    """后台线程：负责加载模型和执行搜索，避免界面卡死"""
    finished = pyqtSignal(list)  # 发送结果列表
    error = pyqtSignal(str)  # 发送错误信息
    status = pyqtSignal(str)  # 发送状态更新

    def __init__(self, query_path, processor, model, device, index, all_paths, top_k, segger, use_yolo):
        super().__init__()
        self.query_path = query_path
        self.processor = processor
        self.model = model
        self.device = device
        self.index = index
        self.all_paths = all_paths
        self.top_k = top_k
        self.segger = segger
        self.use_yolo = use_yolo
    def run(self):
        try:
            self.status.emit("正在提取图像特征...")

            # 1. 预处理
            image = Image.open(self.query_path).convert("RGB")
            # 【关键】先按 EXIF 修正方向，确保 YOLO / 特征提取都基于正确方向
            try:
                image = ImageOps.exif_transpose(image)
            except Exception:
                pass

            # YOLO 裁剪（可选）
            self.status.emit("正在准备输入图像... (YOLO裁剪可选)")
            roi_img, roi_info, _debug_img = apply_yolo_roi(image, self.segger, self.use_yolo)

            # 用裁剪后的 ROI（或原图）提取特征
            inputs = self.processor(images=roi_img, return_tensors="pt").to(self.device)

            # 2. 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                feature = outputs.pooler_output.cpu().numpy()

            # 3. 归一化
            faiss.normalize_L2(feature)

            self.status.emit("正在检索数据库...")

            # 4. 搜索
            distances, indices = self.index.search(feature, self.top_k)

            # 5. 格式化结果
            results = []
            for i, idx in enumerate(indices[0]):
                score = float(distances[0][i])
                # 处理 numpy 类型转换问题
                path = str(self.all_paths[idx])
                results.append({
                    "rank": i + 1,
                    "score": score,
                    "path": path,
                    "filename": os.path.basename(path)
                })

            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class ResultCard(QFrame):
    """单个结果卡片组件"""

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setup_ui()

    def setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            QFrame:hover {
                border: 2px solid #0078D4;
                background-color: #f0f8ff;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 图片标签
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(150, 150)
        self.img_label.setMaximumSize(150, 150)
        self.img_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.load_thumbnail()
        layout.addWidget(self.img_label)

        # 信息标签
        score_color = "#d32f2f" if self.data['score'] > 0.9 else "#1976d2"
        info_text = f"<b>相似度:</b> <span style='color:{score_color}; font-size:14px;'>{self.data['score']:.4f}</span><br>"
        info_text += f"<span style='font-size:10px; color:#555;'>{self.data['filename']}</span>"

        self.info_label = QLabel(info_text)
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

    def load_thumbnail(self):
        try:
            pil = Image.open(self.data['path']).convert("RGB")
            try:
                pil = ImageOps.exif_transpose(pil)
            except Exception:
                pass
            pil.thumbnail((150, 150))
            from PyQt5.QtGui import QImage
            buf = pil.tobytes("raw", "RGB")
            qimg = QImage(buf, pil.width, pil.height, 3 * pil.width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled_pixmap)
        except:
            self.img_label.setText("无法加载\n预览")

    def mouseDoubleClickEvent(self, event):
        """双击打开文件夹并选中文件"""
        path = self.data['path']
        if os.path.exists(path):
            # Windows: 资源管理器选中文件
            if sys.platform == 'win32':
                # 使用 subprocess 调用 explorer /select 更稳定
                subprocess.run(['explorer', '/select,', path])
            else:
                # Mac/Linux
                cmd = ['open', os.path.dirname(path)] if sys.platform == 'darwin' else ['xdg-open',
                                                                                        os.path.dirname(path)]
                subprocess.run(cmd)
        else:
            QMessageBox.warning(self, "错误", f"文件不存在:\n{path}")


class FabricSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = None
        self.model = None
        self.device = None
        self.index = None
        self.all_paths = None
        self.worker = None
        self.segger = None  # ===== [YOLO ADD]
        self.init_ui()
        self.init_backend()

    def init_ui(self):
        self.setWindowTitle("🧵 DINOv3 布料以图搜图系统 (PyQt5)")
        self.setMinimumSize(1000, 700)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === 左侧面板：控制区 ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)

        # 标题
        title = QLabel("🔍 布料搜索")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        left_layout.addSpacing(20)

        # 选择图片按钮
        self.btn_select = QPushButton("📂 选择查询图片")
        self.btn_select.setStyleSheet("""
            QPushButton {
                background-color: #0078D4; color: white; 
                font-size: 16px; padding: 15px; border-radius: 5px;
            }
            QPushButton:hover { background-color: #005a9e; }
        """)
        self.btn_select.clicked.connect(self.select_image)
        left_layout.addWidget(self.btn_select)
        # ===== [YOLO ADD] YOLO 开关 =====
        self.chk_yolo = QPushButton("✅ 启用 YOLO 智能裁剪")
        self.chk_yolo.setCheckable(True)
        self.chk_yolo.setChecked(USE_YOLO_ROI_DEFAULT)
        self.chk_yolo.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; color: white;
                font-size: 13px; padding: 10px; border-radius: 5px;
            }
            QPushButton:checked { background-color: #2e7d32; }
            QPushButton:!checked { background-color: #616161; }
        """)
        self.chk_yolo.setToolTip("关闭：直接用原图检索（更快，可能受背景影响）\n开启：YOLO 分割裁剪（更聚焦主体）")
        left_layout.addWidget(self.chk_yolo)
        # ==============================
        # 查询图预览
        self.query_preview = QLabel("未选择图片")
        self.query_preview.setAlignment(Qt.AlignCenter)
        self.query_preview.setMinimumHeight(200)
        self.query_preview.setStyleSheet("border: 2px dashed #ccc; border-radius: 5px; background: #f9f9f9;")
        left_layout.addWidget(self.query_preview)

        # 状态栏
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        left_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        left_layout.addStretch()

        # 提示信息
        hint = QLabel("💡 提示:\n双击搜索结果可打开所在文件夹。\n相似度 > 0.9 通常为同款。")
        hint.setStyleSheet("color: #888; font-size: 12px;")
        left_layout.addWidget(hint)

        main_layout.addWidget(left_panel)

        # === 右侧面板：结果展示区 ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.results_title = QLabel("搜索结果")
        self.results_title.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(self.results_title)

        # 滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 结果容器 (网格布局)
        self.results_container = QWidget()
        self.grid_layout = QGridLayout(self.results_container)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)

        self.scroll_area.setWidget(self.results_container)
        right_layout.addWidget(self.scroll_area)

        main_layout.addWidget(right_panel, 1)  # 拉伸因子为1

    def init_backend(self):
        """初始化模型和索引"""
        self.status_label.setText("正在加载模型和索引... (首次可能较慢)")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 无限循环

        try:
            # 加载模型
            self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModel.from_pretrained(MODEL_ID).eval().to(self.device)

            # 加载索引
            if not os.path.exists(INDEX_PATH) or not os.path.exists(PATHS_FILE):
                raise FileNotFoundError(f"找不到索引文件: {INDEX_PATH}")

            self.index = faiss.read_index(INDEX_PATH)
            self.all_paths = np.load(PATHS_FILE, allow_pickle=True)

            # ===== [YOLO ADD] 初始化 YOLO（可用就加载，失败不影响主流程）=====
            if YOLO_AVAILABLE:
                try:
                    self.status_label.setText("正在加载 YOLO 模型...（可选功能）")
                    QApplication.processEvents()
                    self.segger = SegPreprocessor(
                        yolo_model_path=YOLO_MODEL_PATH,
                        yolo_device=YOLO_DEVICE
                    )
                except Exception as e:
                    print(f"❌ YOLO 加载失败: {e}")
                    self.segger = None
            # ===========================================================
            yolo_txt = "可用" if self.segger else ("不可用" if YOLO_AVAILABLE else "未安装")
            self.status_label.setText(
                f"✅ 就绪 | 库容量: {len(self.all_paths)} 张 | 设备: {self.device.upper()} | YOLO: {yolo_txt}")
            self.progress_bar.setVisible(False)

        except Exception as e:
            self.status_label.setText(f"❌ 初始化失败: {e}")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "错误", f"无法加载模型或索引:\n{e}\n\n请检查配置文件和文件路径。")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择布料图片", "", "Images (*.jpg *.png *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.run_search(file_path)

    def run_search(self, image_path):
        # 更新左侧预览
        # ===== [YOLO ADD] 查询预览：EXIF修正 + 可选显示YOLO框 =====
        use_yolo_flag = self.chk_yolo.isChecked()

        try:
            pil = Image.open(image_path).convert("RGB")
            pil = ImageOps.exif_transpose(pil)
        except Exception:
            pil = None

        if pil is not None:
            # 预览上画框（如果开启YOLO且可用）
            _roi, _info, debug_img = apply_yolo_roi(pil, self.segger, use_yolo_flag)
            # PIL -> QPixmap
            debug_img = debug_img.convert("RGB")
            w, h = debug_img.size
            # 更直接的方式：先转 bytes 构造 QImage
            from PyQt5.QtGui import QImage
            buf = debug_img.tobytes("raw", "RGB")
            qimage = QImage(buf, w, h, 3 * w, QImage.Format_RGB888).copy()  # ✅ 强制拷贝，避免闪退
            pixmap = QPixmap.fromImage(qimage).scaled(280, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            pixmap = QPixmap(image_path).scaled(280, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.query_preview.setPixmap(pixmap)
        # ==========================================================

        # 清空旧结果
        self.clear_results()
        mode_str = "YOLO裁剪" if use_yolo_flag else "原图"
        self.results_title.setText(f"搜索中... ({mode_str})")

        # 启动后台线程
        self.worker = SearchWorker(
            image_path, self.processor, self.model, self.device,
            self.index, self.all_paths, TOP_K,
            self.segger, use_yolo_flag
        )
        self.worker.status.connect(lambda s: self.status_label.setText(f"⏳ {s}"))
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()

    def clear_results(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def display_results(self, results):
        self.results_title.setText(f"✅ 找到 {len(results)} 个相似结果")
        self.status_label.setText("搜索完成")

        if not results:
            lbl = QLabel("未找到匹配结果")
            lbl.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(lbl, 0, 0)
            return

        # 网格布局填充 (每行 3 个)
        cols = 3
        for i, data in enumerate(results):
            row = i // cols
            col = i % cols

            card = ResultCard(data)
            card.setMinimumWidth(180)
            self.grid_layout.addWidget(card, row, col)

        # 添加弹性空间到底部，防止卡片挤在顶部
        self.grid_layout.setRowStretch(self.grid_layout.rowCount(), 1)

    def show_error(self, msg):
        self.status_label.setText("❌ 搜索失败")
        QMessageBox.critical(self, "搜索错误", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局样式 (可选，让界面更现代)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(245, 245, 245))
    app.setPalette(palette)

    window = FabricSearchApp()
    window.show()
    sys.exit(app.exec_())