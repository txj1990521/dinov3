# seg_preprocess.py
# -*- coding: utf-8 -*-
"""
YOLO-seg based foreground extraction + background fill (KEEP SAME SIZE).

Usage:
    from seg_preprocess import SegPreprocessor, SegConfig

    segger = SegPreprocessor()
    pil_clean, mask_u8 = segger.process_pil(pil_img, cfg)

Key points:
- Output image keeps the same HxW as input (no crop/rotate).
- Returns full-size mask_u8 (0/255).
- YOLO model is cached (loaded once).
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Callable

import cv2
import numpy as np
from PIL import Image

# -----------------------------
# Optional: YOLO dependencies
# You already have ultralytics YOLO in your project
# -----------------------------
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# =========================
# Config
# =========================
@dataclass
class SegConfig:
    # main thresholds
    score_thr: float = 0.6
    min_area_frac: float = 0.06

    # class filtering
    use_classes: Optional[List[int]] = None
    merge_all: bool = True

    # bg fill mode
    bg_mode: str = "mean"  # "mean" | "white" | "edge"

    # morphology
    close_ksize: int = 11
    close_iter: int = 2
    open_iter: int = 1

    # yolo predict params
    yolo_imgsz: int = 640
    yolo_iou: float = 0.5
    yolo_retina_masks: bool = True
    max_det: int = 100

    # runtime
    device: Union[str, int] = 0  # 0/1... or "cpu" or "cuda"

    # debug
    debug_dir: Optional[str] = None
    debug_tag_prefix: str = "seg"


# =========================
# Helpers: PIL <-> BGR
# =========================
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def ensure_dir(d: str):
    if d:
        os.makedirs(d, exist_ok=True)

def _largest_cc(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    """Keep largest connected component in binary mask."""
    if mask_u8 is None or mask_u8.size == 0:
        return None
    m = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return mask_u8
    # stats: [label, x, y, w, h, area] with background at 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    out = (labels == k).astype(np.uint8) * 255
    return out

def _fill_background(img_bgr: np.ndarray, mask_u8: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    mode:
      - mean: fill with mean color of foreground (masked region)
      - white: fill with white
      - edge:  fill with blurred edge (cheap)
    """
    mode = (mode or "mean").lower()
    H, W = img_bgr.shape[:2]
    out = img_bgr.copy()

    fg = (mask_u8 > 0)
    bg = ~fg

    if mode == "white":
        out[bg] = (255, 255, 255)
        return out

    if mode == "edge":
        # blur image and use bg from blurred
        blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=12)
        out[bg] = blur[bg]
        return out

    # default: mean
    if fg.sum() < 10:
        # not enough fg: fallback to edge
        blur = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=12)
        out[bg] = blur[bg]
        return out

    mean_color = img_bgr[fg].mean(axis=0)
    out[bg] = mean_color
    return out


# =========================
# YOLO Seg wrapper (cached)
# =========================
_YOLO_MODEL_CACHE = {}

def get_yolo_seg(model_path: Optional[str] = None):
    """
    Return cached YOLO model.
    - If model_path is None: user must have default in their environment.
    - For your project, you likely already have a get_yolo_seg() elsewhere.
      You can replace this function with your own import OR pass model in SegPreprocessor.
    """
    key = model_path or "__default__"
    if key in _YOLO_MODEL_CACHE:
        return _YOLO_MODEL_CACHE[key]
    if YOLO is None:
        raise ImportError("ultralytics is not available. Please install ultralytics or provide your own YOLO model.")
    if model_path is None:
        # You can set a default weights file here if you want:
        # model_path = "yolo11x-seg.pt" ...
        raise ValueError("model_path is None. Please set SegPreprocessor(yolo_model_path=...) or edit get_yolo_seg().")
    model = YOLO(model_path)
    _YOLO_MODEL_CACHE[key] = model
    return model


def _yolo_extract_mask_u8(
    r0,
    H: int,
    W: int,
    conf_thr: float,
    use_classes: Optional[List[int]],
    merge_all: bool = True,
) -> Optional[np.ndarray]:
    """
    Extract union mask from ultralytics result.
    Returns uint8 mask (0/255) in original HxW.

    Notes:
    - This matches common ultralytics segmentation output:
      r0.masks.data: (N, h, w) float/bool
      r0.boxes.cls, r0.boxes.conf: class/conf
    - If your project already has _yolo_extract_mask_u8, you can replace this with your own.
    """
    if r0 is None or r0.masks is None:
        return None

    masks = r0.masks.data  # torch tensor (N, mh, mw)
    if masks is None:
        return None

    # boxes info
    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return None

    cls = boxes.cls
    conf = boxes.conf
    if cls is None or conf is None:
        return None

    cls = cls.detach().cpu().numpy().astype(np.int32)
    conf = conf.detach().cpu().numpy().astype(np.float32)

    keep = conf >= float(conf_thr)
    if use_classes is not None:
        use_set = set(int(x) for x in use_classes)
        keep = keep & np.array([c in use_set for c in cls], dtype=bool)

    idx = np.where(keep)[0]
    if idx.size == 0:
        return None

    # merge masks
    m = masks[idx]  # (K, mh, mw)
    m = m.detach().float().cpu().numpy()  # float in [0,1]
    m = (m > 0.5).astype(np.uint8) * 255

    if merge_all:
        u = np.max(m, axis=0)
    else:
        # keep the largest mask by area
        areas = m.reshape(m.shape[0], -1).sum(axis=1)
        k = int(np.argmax(areas))
        u = m[k]

    # resize to original (H,W) if needed
    mh, mw = u.shape[:2]
    if (mh, mw) != (H, W):
        u = cv2.resize(u, (W, H), interpolation=cv2.INTER_NEAREST)
    return u


# =========================
# Main preprocessor class
# =========================
class SegPreprocessor:
    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        yolo_model=None,
        yolo_getter: Optional[Callable] = None,
        yolo_device: Union[str, int] = 0,
    ):
        """
        Provide ONE of:
          - yolo_model (already loaded)
          - yolo_model_path
          - yolo_getter() that returns a model

        yolo_device: passed into model.predict(device=...)
        """
        self.yolo_model_path = yolo_model_path
        self.yolo_device = yolo_device

        if yolo_model is not None:
            self.model = yolo_model
        elif yolo_getter is not None:
            self.model = yolo_getter()
        else:
            # lazy load when first used
            self.model = None

    def _get_model(self):
        if self.model is not None:
            return self.model
        self.model = get_yolo_seg(self.yolo_model_path)
        return self.model

    def process_bgr(self, img_bgr: np.ndarray, cfg: SegConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          out_bgr  : same size as input, background filled
          mask_u8  : same size as input, 0/255
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr, None

        H, W = img_bgr.shape[:2]
        default_mask = np.ones((H, W), dtype=np.uint8) * 255

        model = self._get_model()

        results = model.predict(
            source=img_bgr,
            conf=float(cfg.score_thr),
            iou=float(cfg.yolo_iou),
            imgsz=int(cfg.yolo_imgsz),
            device=cfg.device if cfg.device is not None else self.yolo_device,
            max_det=int(cfg.max_det),
            retina_masks=bool(cfg.yolo_retina_masks),
            classes=cfg.use_classes,
            verbose=False,
            stream=False,
            save=False,
            show=False
        )

        if results is None or len(results) == 0:
            return img_bgr, default_mask

        r0 = results[0]
        mask_u8 = _yolo_extract_mask_u8(
            r0, H, W,
            conf_thr=float(cfg.score_thr),
            use_classes=cfg.use_classes,
            merge_all=bool(cfg.merge_all)
        )
        if mask_u8 is None:
            return img_bgr, default_mask

        # cleanup mask
        k = int(cfg.close_ksize)
        k = k if (k % 2 == 1) else (k + 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, ker, iterations=int(cfg.close_iter))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, ker, iterations=int(cfg.open_iter))

        mask_u8 = _largest_cc(mask_u8)
        if mask_u8 is None:
            return img_bgr, default_mask

        area = float((mask_u8 > 0).sum())
        if area < float(cfg.min_area_frac) * float(H * W):
            return img_bgr, default_mask

        # fill bg (keep size)
        filled = _fill_background(img_bgr, mask_u8, mode=str(cfg.bg_mode))
        out = filled.copy()
        fg = (mask_u8 > 0)
        out[fg] = img_bgr[fg]

        # debug
        if cfg.debug_dir:
            ensure_dir(cfg.debug_dir)
            tag = f"{cfg.debug_tag_prefix}_{int(time.time()*1000)}"
            cv2.imwrite(os.path.join(cfg.debug_dir, f"{tag}_mask.png"), mask_u8)
            cv2.imwrite(os.path.join(cfg.debug_dir, f"{tag}_out.png"), out)

        return out, mask_u8

    def process_pil(self, pil_img: Image.Image, cfg: SegConfig) -> Tuple[Image.Image, np.ndarray]:
        bgr = pil_to_bgr(pil_img)
        out_bgr, mask_u8 = self.process_bgr(bgr, cfg)
        return bgr_to_pil(out_bgr), mask_u8
