from __future__ import annotations

import glob
import os
from typing import Optional, Union

import numpy as np
from PIL import Image
import cv2
import torch

from inference import get_mask
from train import AnimeSegmentation, net_names

# Simple global cache (avoid redundant loads)
_cached_model = None
_cached_device = None
_cached_img_size = None


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


def _ensure_model(
    model: Optional[torch.nn.Module],
    net: str,
    ckpt: Optional[str],
    device: Optional[str],
    img_size: int,
) -> torch.nn.Module:
    """Prepare and return the model. If model is provided, use it. Otherwise load ckpt.
    Once loaded, use the global cache."""
    global _cached_model, _cached_device, _cached_img_size

    if model is not None:
        return model

    if _cached_model is not None and _cached_device == device and _cached_img_size == img_size:
        return _cached_model

    if ckpt is None:
        # Auto-discover (first found .ckpt)
        ckpts = sorted(glob.glob("**/*.ckpt", recursive=True))
        if not ckpts:
            raise FileNotFoundError("No .ckpt found. Please specify ckpt path.")
        ckpt = ckpts[0]

    # Device fallback
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    if net not in net_names:
        raise ValueError(f"Unsupported net: {net}. choices={net_names}")

    m = AnimeSegmentation.try_load(net, ckpt, device, img_size)
    m.eval()
    m.to(device)

    _cached_model = m
    _cached_device = device
    _cached_img_size = img_size
    return m


def convert_img(
    img: Union[Image.Image, np.ndarray, str],
    *,
    model: Optional[torch.nn.Module] = None,
    net: str = "isnet_is",
    ckpt: Optional[str] = None,
    device: Optional[str] = None,
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
      - device: cpu / cuda:0 (auto-detect if not provided; fallback to CPU if CUDA unavailable)
      - img_size: inference size
      - bg_white: if True, composite over white background; if False, keep transparent background.

    Returns:
      - PIL.Image (mode="RGBA")
    """
    rgb = _to_rgb_ndarray(img)

    # Prepare model
    mdl = _ensure_model(model, net=net, ckpt=ckpt, device=device, img_size=img_size)

    # Predicted mask [H, W, 1], values in [0, 1]
    # Use AMP only when CUDA is available
    use_amp = mdl.device.type == "cuda" and torch.cuda.is_available()
    mask = get_mask(mdl, rgb, use_amp=use_amp, s=img_size)

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
