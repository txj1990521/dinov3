import os
import math
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import faiss

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QSpinBox, QScrollArea, QMessageBox, QLineEdit, QGroupBox
)

from dinov3.models.vision_transformer import vit_large

# =========================
# CONFIG
# =========================
CKPT = r"D:/zhanlanProject/dinov3/pre_model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
OUT_DIR = r"D:\zhanlan\faiss_database_dinov3_hybrid"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GLOBAL_INDEX = os.path.join(OUT_DIR, "global.index")
GLOBAL_META  = os.path.join(OUT_DIR, "global_img_paths.npy")

DEFAULT_TOPK = 12
DEFAULT_COLS = 4

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# =========================
# Core (same as your logic)
# =========================
def build_dinov3_vitl16(ckpt_path: str, device: str):
    model = vit_large(patch_size=16)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    cleaned = {}
    for k, v in sd.items():
        kk = k
        for pref in ("module.", "model.", "backbone."):
            if kk.startswith(pref):
                kk = kk[len(pref):]
        cleaned[kk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[MODEL] loaded. missing={len(missing)} unexpected={len(unexpected)}")

    model.eval().to(device)
    return model


def imread_unicode(p):
    data = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


@torch.no_grad()
def extract_embedding(model, img_tensor):
    feats = model.forward_features(img_tensor)

    if isinstance(feats, dict):
        for k in ["x_norm_clstoken", "cls_token", "clstoken", "x_cls", "feat_cls"]:
            if k in feats and torch.is_tensor(feats[k]):
                cls = feats[k]
                return F.normalize(cls, dim=1)

        for k in ["x_norm", "tokens", "x", "last_hidden_state", "x_prenorm"]:
            if k in feats and torch.is_tensor(feats[k]):
                tok = feats[k]
                if tok.dim() == 3:
                    cls = tok[:, 0, :]
                    return F.normalize(cls, dim=1)
                if tok.dim() == 2:
                    return F.normalize(tok, dim=1)

        raise KeyError(f"forward_features returned dict but no known keys found. keys={list(feats.keys())}")

    if torch.is_tensor(feats):
        if feats.dim() == 3:
            cls = feats[:, 0, :]
        elif feats.dim() == 2:
            cls = feats
        else:
            raise ValueError(f"Unexpected feats tensor shape: {feats.shape}")
        return F.normalize(cls, dim=1)

    raise TypeError(f"Unexpected forward_features type: {type(feats)}")


def load_faiss(index_path, meta_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    index = faiss.read_index(index_path)
    img_paths = np.load(meta_path, allow_pickle=True)
    return index, img_paths


def search_one(model, index, img_paths, query_img_path, topk=10):
    img = imread_unicode(query_img_path)
    if img is None:
        raise ValueError(f"Cannot read query image: {query_img_path}")

    q = preprocess(img).unsqueeze(0).to(DEVICE)
    q_feat = extract_embedding(model, q).cpu().numpy().astype("float32")

    scores, ids = index.search(q_feat, topk)

    results = []
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if idx < 0:
            continue
        results.append((rank, float(score), str(img_paths[idx])))
    return results


# =========================
# Montage helpers (same style)
# =========================
def _fit_to_box(img_bgr, box_w, box_h):
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((box_h, box_w, 3), np.uint8)

    scale = min(box_w / w, box_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((box_h, box_w, 3), np.uint8)
    x0 = (box_w - nw) // 2
    y0 = (box_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def make_retrieval_montage(query_path, results, out_path,
                          cols=4, cell_w=260, cell_h=360,
                          margin=30, header_h=60,
                          font_scale=1.2, thickness=3):
    items = [("QUERY", None, query_path)]
    for rank, score, path in results:
        items.append((f"#{rank}", float(score), path))

    n = len(items)
    rows = math.ceil(n / cols)

    W = margin + cols * (cell_w + margin)
    H = margin + rows * (cell_h + header_h + margin)

    canvas = np.zeros((H, W, 3), np.uint8)

    for idx, (title, score, path) in enumerate(items):
        r = idx // cols
        c = idx % cols

        x = margin + c * (cell_w + margin)
        y = margin + r * (cell_h + header_h + margin)

        img = imread_unicode(path)
        if img is None:
            img = np.zeros((cell_h, cell_w, 3), np.uint8)

        tile = _fit_to_box(img, cell_w, cell_h)
        canvas[y + header_h:y + header_h + cell_h, x:x + cell_w] = tile

        text = "QUERY" if title == "QUERY" else f"{title}  {score:.3f}"
        cv2.putText(
            canvas, text,
            (x, y + int(header_h * 0.75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), thickness,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(out_path, canvas)
    return out_path


# =========================
# Qt helpers
# =========================
def bgr_to_qpixmap(img_bgr, max_w=None, max_h=None):
    if img_bgr is None:
        return QPixmap()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    if max_w is not None or max_h is not None:
        scale = 1.0
        if max_w is not None:
            scale = min(scale, max_w / w)
        if max_h is not None:
            scale = min(scale, max_h / h)
        if scale < 1.0:
            img_rgb = cv2.resize(img_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
            h, w = img_rgb.shape[:2]

    qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ResultCard(QWidget):
    def __init__(self, rank, score, path, thumb_w=220, thumb_h=220):
        super().__init__()
        self.path = path

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        img = imread_unicode(path)
        pix = bgr_to_qpixmap(img, max_w=thumb_w, max_h=thumb_h)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setPixmap(pix)
        img_label.setStyleSheet("background-color: black;")
        layout.addWidget(img_label)

        text = QLabel(f"#{rank}  {score:.3f}\n{path}")
        text.setWordWrap(True)
        text.setStyleSheet("color: white;")
        font = QFont()
        font.setPointSize(9)
        text.setFont(font)
        layout.addWidget(text)

        self.setLayout(layout)
        self.setStyleSheet("""
            QWidget {
                background-color: #111;
                border: 1px solid #333;
                border-radius: 8px;
            }
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DINOv3 + FAISS Image Retrieval")
        self.resize(1280, 800)

        # load resources once
        self.index = None
        self.img_paths = None
        self.model = None

        self.query_path = None
        self.last_results = []

        self._build_ui()
        self._load_all()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        # Left panel (query)
        left = QVBoxLayout()
        left.setSpacing(10)

        grp_query = QGroupBox("Query")
        ql = QVBoxLayout(grp_query)

        self.query_preview = QLabel("请选择一张Query图片")
        self.query_preview.setAlignment(Qt.AlignCenter)
        self.query_preview.setMinimumSize(320, 320)
        self.query_preview.setStyleSheet("background-color:black;color:white;border:1px solid #333;border-radius:8px;")
        ql.addWidget(self.query_preview)

        btn_row = QHBoxLayout()
        self.btn_pick = QPushButton("选择图片")
        self.btn_search = QPushButton("检索")
        self.btn_montage = QPushButton("导出拼图")
        btn_row.addWidget(self.btn_pick)
        btn_row.addWidget(self.btn_search)
        btn_row.addWidget(self.btn_montage)
        ql.addLayout(btn_row)

        opts = QHBoxLayout()
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 200)
        self.topk_spin.setValue(DEFAULT_TOPK)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 10)
        self.cols_spin.setValue(DEFAULT_COLS)
        opts.addWidget(QLabel("TopK:"))
        opts.addWidget(self.topk_spin)
        opts.addSpacing(10)
        opts.addWidget(QLabel("Cols:"))
        opts.addWidget(self.cols_spin)
        ql.addLayout(opts)

        self.status_line = QLineEdit()
        self.status_line.setReadOnly(True)
        self.status_line.setText("Ready.")
        ql.addWidget(self.status_line)

        left.addWidget(grp_query)

        # Right panel (results with scroll)
        grp_res = QGroupBox("Results")
        rr = QVBoxLayout(grp_res)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color:black;border:1px solid #333;border-radius:8px;")

        self.res_container = QWidget()
        self.grid = QGridLayout(self.res_container)
        self.grid.setContentsMargins(12, 12, 12, 12)
        self.grid.setSpacing(12)

        self.scroll.setWidget(self.res_container)
        rr.addWidget(self.scroll)

        # add to root
        root_layout.addLayout(left, 1)
        root_layout.addWidget(grp_res, 3)

        # connect
        self.btn_pick.clicked.connect(self.on_pick)
        self.btn_search.clicked.connect(self.on_search)
        self.btn_montage.clicked.connect(self.on_export_montage)

        # dark theme-ish
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0b0b; }
            QLabel, QGroupBox { color: white; }
            QPushButton { padding: 8px 12px; border-radius: 8px; background:#222; color:white; border:1px solid #444; }
            QPushButton:hover { background:#2a2a2a; }
            QLineEdit { background:#111; color:white; border:1px solid #333; border-radius: 6px; padding: 6px; }
            QSpinBox { background:#111; color:white; border:1px solid #333; border-radius: 6px; padding: 4px; }
            QGroupBox { border:1px solid #333; border-radius: 8px; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        """)

    def _load_all(self):
        try:
            self.status_line.setText("Loading FAISS index...")
            self.index, self.img_paths = load_faiss(GLOBAL_INDEX, GLOBAL_META)
            self.status_line.setText(f"Index loaded. ntotal={self.index.ntotal}")

            self.status_line.setText("Loading model...")
            self.model = build_dinov3_vitl16(CKPT, DEVICE)
            self.status_line.setText(f"Model loaded on {DEVICE}. Ready.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self.status_line.setText("Load failed.")

    def on_pick(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择Query图片", "", f"Images (*{' *'.join(IMG_EXTS)})"
        )
        if not path:
            return
        self.query_path = path
        img = imread_unicode(path)
        pix = bgr_to_qpixmap(img, max_w=520, max_h=520)
        self.query_preview.setPixmap(pix)
        self.status_line.setText(f"Selected: {path}")

    def _clear_grid(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def on_search(self):
        if self.model is None or self.index is None:
            QMessageBox.warning(self, "Not Ready", "Model/index not loaded.")
            return
        if not self.query_path or not os.path.exists(self.query_path):
            QMessageBox.warning(self, "No Query", "请先选择一张Query图片。")
            return

        try:
            topk = int(self.topk_spin.value())
            cols = int(self.cols_spin.value())

            self.status_line.setText("Searching...")
            results = search_one(self.model, self.index, self.img_paths, self.query_path, topk=topk)
            self.last_results = results

            self._clear_grid()
            for i, (rank, score, path) in enumerate(results):
                card = ResultCard(rank, score, path, thumb_w=240, thumb_h=240)
                r = i // cols
                c = i % cols
                self.grid.addWidget(card, r, c)

            self.status_line.setText(f"Done. Found {len(results)} results.")
        except Exception as e:
            QMessageBox.critical(self, "Search Error", str(e))
            self.status_line.setText("Search failed.")

    def on_export_montage(self):
        if not self.query_path or not self.last_results:
            QMessageBox.information(self, "No Results", "请先选择图片并检索，再导出拼图。")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存拼图", os.path.join(OUT_DIR, "montage.png"), "PNG (*.png)"
        )
        if not save_path:
            return

        try:
            cols = int(self.cols_spin.value())
            out = make_retrieval_montage(
                query_path=self.query_path,
                results=self.last_results,
                out_path=save_path,
                cols=cols,
                cell_w=260,
                cell_h=360,
                margin=30,
                header_h=60
            )
            QMessageBox.information(self, "Saved", f"拼图已保存：\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
