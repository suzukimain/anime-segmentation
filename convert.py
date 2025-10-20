from __future__ import annotations

import os
from typing import Union

import numpy as np
from PIL import Image
import cv2
import torch

from inference import get_mask


def _to_rgb_ndarray(img: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
    """
    Normalize image input to an RGB numpy.ndarray [H, W, 3] (uint8).

    - PIL.Image: convert to RGB considering modes like RGB/BGR/LA/L
    - numpy.ndarray: if HxWx3 is given, treat it as RGB (do not attempt BGR->RGB).
      If the array likely comes from OpenCV (BGR), call cv2.cvtColor(...) in the caller as needed.
    - str: treat as a path (cv2.imread returns BGR) -> convert to RGB
    """
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    if isinstance(img, np.ndarray):
        if img.ndim == 2:  # grayscale
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 3:
            # The received array may be RGB or BGR; here we treat it as RGB.
            # If needed, convert BGR->RGB in the caller using cv2.cvtColor(..., COLOR_BGR2RGB).
            return img.astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 4:
            # RGBA -> RGB (alpha will be re-estimated later, so discard)
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        raise ValueError("Unsupported numpy image shape. Expected HxW, HxWx3, or HxWx4.")

    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image path not found: {img}")
        bgr = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise ValueError(f"Failed to read image: {img}")
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if bgr.shape[2] == 4:
            # BGRA -> BGR -> RGB
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    raise TypeError("img must be PIL.Image, numpy.ndarray, or path str")


def convert_img(
    img: Union[Image.Image, np.ndarray, str],
    *,
    model: torch.nn.Module,
    img_size: int = 1024,
    bg_white: bool = False,
) -> Image.Image:
    """
    Return a PIL.Image with the input image background removed (transparent) or composited
    over white.

    Args:
      - img: PIL.Image / numpy.ndarray / image path (str)
      - model: already loaded model (optional)
      - net: network name to use when loading ckpt
      - ckpt: model checkpoint path (auto-discover if not provided)
        -  model = AnimeSegmentation.from_pretrained("skytnt/anime-seg").to(device)
      - device: cpu / cuda:0 (auto-detect if not provided; fallback to CPU if CUDA unavailable)
      - img_size: inference size
      - bg_white: if True, composite over white background; if False, keep transparent background.

    Returns:
      - PIL.Image (mode="RGBA")
    """
    rgb = _to_rgb_ndarray(img)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare model
    use_amp = device == "cuda" and torch.cuda.is_available()
    mask = get_mask(model, rgb, use_amp=use_amp, s=img_size)

    # Alpha composition: produce RGBA
    h, w = rgb.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h))

    if bg_white:
        # White background: out_rgb = mask*rgb + (1-mask)*255
        out_rgb = (mask * rgb + (1 - mask) * 255).clip(0, 255).astype(np.uint8)
        alpha = (mask * 255).astype(np.uint8)
        out = np.concatenate([out_rgb, alpha], axis=2)
    else:
        # Transparent background: keep RGB, alpha = mask
        alpha = (mask * 255).astype(np.uint8)
        out = np.concatenate([rgb, alpha], axis=2)

    return Image.fromarray(out, mode="RGBA")


__all__ = ["convert_img"]
