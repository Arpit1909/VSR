# -*- coding: utf-8 -*-
"""
E2FGVI-HQ Video Inpainting Wrapper
===================================
Drop-in replacement for VideoInpaint (ProPainter).
Same interface: E2FGVIInpaint.inpaint(frames, mask) → comp_frames

Setup required (one-time):
  1. Clone repo:
       git clone https://github.com/MCG-NKU/E2FGVI backend/inpaint/e2fgvi_repo
  2. Download weights to backend/models/e2fgvi/E2FGVI-HQ-CVPR22.pth
       https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3
"""

import os
import sys
import cv2
import numpy as np
import torch
import warnings
import importlib.util

from PIL import Image
from torchvision import transforms

from backend import config
from backend.inpaint.video.model.misc import get_device

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
E2FGVI_REPO = os.path.join(config.BASE_DIR, 'inpaint', 'e2fgvi_repo')
E2FGVI_MODEL_PATH = os.path.join(config.BASE_DIR, 'models', 'e2fgvi', 'E2FGVI-HQ-CVPR22.pth')


def _ensure_repo_on_path():
    """Add the cloned E2FGVI repo to sys.path so its model package is importable."""
    if not os.path.isdir(E2FGVI_REPO):
        raise RuntimeError(
            f"\n[E2FGVI-HQ] Repository not found at:\n  {E2FGVI_REPO}\n\n"
            "Please run this command from your project root:\n"
            f"  git clone https://github.com/MCG-NKU/E2FGVI \"{E2FGVI_REPO}\"\n"
        )
    if E2FGVI_REPO not in sys.path:
        sys.path.insert(0, E2FGVI_REPO)


def _load_inpaint_generator_class_from_repo():
    """
    Load E2FGVI-HQ InpaintGenerator from the *exact* repo path.

    Important: importing `from model...` is fragile because many Python projects
    may have a top-level `model` package on sys.path. Loading by file path
    guarantees we use the intended cloned repo implementation (including the
    mmcv-free modules shipped in this repo).
    """
    if not bool(getattr(config, "E2FGVI_USE_HQ", True)):
        raise RuntimeError(
            "[E2FGVI-HQ] Non-HQ E2FGVI (model/e2fgvi.py) is not supported in this build. "
            "Keep E2FGVI_USE_HQ = True in backend/config.py."
        )
    e2fgvi_hq_py = os.path.join(E2FGVI_REPO, "model", "e2fgvi_hq.py")
    if not os.path.isfile(e2fgvi_hq_py):
        raise FileNotFoundError(
            f"\n[E2FGVI-HQ] Missing file:\n  {e2fgvi_hq_py}\n"
            "Your E2FGVI repo looks incomplete. Re-run:\n"
            f"  git clone https://github.com/MCG-NKU/E2FGVI \"{E2FGVI_REPO}\"\n"
        )

    # Ensure relative imports inside the repo resolve to *this* repo.
    _ensure_repo_on_path()

    spec = importlib.util.spec_from_file_location("_vsr_e2fgvi_hq", e2fgvi_hq_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"[E2FGVI-HQ] Could not load module spec for: {e2fgvi_hq_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "InpaintGenerator"):
        raise ImportError(f"[E2FGVI-HQ] InpaintGenerator not found in: {e2fgvi_hq_py}")
    return mod.InpaintGenerator


def _get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    """
    Identical logic to ProPainter's get_ref_index.
    Collects global reference frames evenly spaced across the video.
    """
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx   = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) >= ref_num:
                    break
                ref_index.append(i)
    return ref_index


def _pad_to_model_size(tensor):
    """
    Pad H and W dims of a [B, T, C, H, W] tensor to multiples of 60 (H) and 108 (W).
    These are the same modulo values used in the official E2FGVI-HQ test.py.
    Returns (padded_tensor, (pad_h, pad_w)).
    """
    h, w = tensor.shape[-2], tensor.shape[-1]
    pad_h = (60 - h % 60) % 60
    pad_w = (108 - w % 108) % 108
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0)
    # NOTE: torch.nn.functional.pad(..., mode='reflect') is unreliable on 5D tensors
    # across PyTorch builds. Flatten to 4D (BT, C, H, W), pad, then reshape back.
    try:
        b, t, c, _, _ = tensor.shape
    except Exception:
        # Fallback: attempt direct pad (should rarely happen)
        padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return padded, (pad_h, pad_w)

    bt = b * t
    x4 = tensor.contiguous().view(bt, c, h, w)
    x4p = torch.nn.functional.pad(x4, (0, pad_w, 0, pad_h), mode='reflect')
    hp, wp = x4p.shape[-2], x4p.shape[-1]
    padded = x4p.view(b, t, c, hp, wp)
    return padded, (pad_h, pad_w)


def _mask_bbox(mask_gray: np.ndarray):
    """Return (y0, y1, x0, x1) bbox of non-zero mask, or None."""
    if mask_gray is None:
        return None
    if mask_gray.ndim != 2:
        return None
    ys, xs = np.where(mask_gray > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return y0, y1, x0, x1


def _clamp(v, lo, hi):
    return max(lo, min(int(v), int(hi)))


class E2FGVIInpaint:
    """
    E2FGVI-HQ video inpainting.

    Key advantages over ProPainter:
      - Single end-to-end model (no separate RAFT + FlowCompleteNet)
      - Flow estimation and inpainting trained jointly → consistent fills
      - No hard chunk-boundary flickering at large neighbor_length values
      - One model file (~200 MB) vs three ProPainter files (~500 MB total)

    Parameters
    ----------
    neighbor_length : int
        Temporal window size. Higher = better context but more VRAM.
        E2FGVI_NEIGHBOR_LENGTH in config.py controls this.
    ref_stride : int
        Stride for global reference frames. Lower = denser global context.
        E2FGVI_REF_STRIDE in config.py controls this.
    """

    def __init__(self, neighbor_length: int = None, ref_stride: int = None):
        self.device         = get_device()
        self.neighbor_length = neighbor_length or int(getattr(config, 'E2FGVI_NEIGHBOR_LENGTH', 10))
        self.ref_stride      = ref_stride      or int(getattr(config, 'E2FGVI_REF_STRIDE', 10))
        self.model           = self._load_model()
        self._model_is_half  = False

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            InpaintGenerator = _load_inpaint_generator_class_from_repo()
        except Exception as exc:
            raise ImportError(
                f"\n[E2FGVI-HQ] Cannot import InpaintGenerator: {exc}\n"
                f"Make sure the repo is correctly cloned at:\n  {E2FGVI_REPO}\n"
                "If you previously cloned E2FGVI elsewhere, remove it to avoid name clashes.\n"
            ) from exc

        if not os.path.exists(E2FGVI_MODEL_PATH):
            raise FileNotFoundError(
                f"\n[E2FGVI-HQ] Model weights not found at:\n  {E2FGVI_MODEL_PATH}\n\n"
                "Download instructions:\n"
                "  1. Visit: https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3\n"
                "  2. Save as:  backend/models/e2fgvi/E2FGVI-HQ-CVPR22.pth\n"
            )

        model = InpaintGenerator()

        state = torch.load(E2FGVI_MODEL_PATH, map_location='cpu')
        # Checkpoints may be saved as {'netG': state_dict} or bare state_dict
        if isinstance(state, dict):
            if 'netG' in state:
                model.load_state_dict(state['netG'])
            elif 'state_dict' in state:
                model.load_state_dict(state['state_dict'])
            else:
                model.load_state_dict(state)
        else:
            model.load_state_dict(state)

        model.to(self.device).eval()
        print(f'[E2FGVI-HQ] Model loaded  →  device={self.device}  '
              f'neighbor_length={self.neighbor_length}  ref_stride={self.ref_stride}  '
              f'weights={os.path.basename(E2FGVI_MODEL_PATH)}')
        return model

    # ── Main inference entry point ─────────────────────────────────────────────

    def inpaint(self, frames, mask):
        """
        Parameters
        ----------
        frames : list[np.ndarray]  — BGR uint8, shape (H, W, 3)
                 OR list[PIL.Image] — RGB
        mask   : np.ndarray (H, W) or (H, W, 3)
                 White (255) = region to fill.  Black (0) = keep original.

        Returns
        -------
        list[np.ndarray]  — BGR uint8, same spatial size as input frames.
        """
        # ── 1. Normalise inputs ────────────────────────────────────────────────
        if isinstance(frames[0], np.ndarray):
            pil_frames      = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
            ori_frames_rgb  = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        else:
            pil_frames      = frames
            ori_frames_rgb  = [np.array(f) for f in frames]

        if isinstance(mask, np.ndarray):
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
        else:
            mask_gray = np.array(mask.convert('L'))

        video_length    = len(pil_frames)
        orig_h, orig_w  = ori_frames_rgb[0].shape[:2]

        # ── 1.5 Crop to mask bbox to reduce VRAM ───────────────────────────────
        # Full-frame 1080p E2FGVI is extremely VRAM heavy. Since subtitle removal
        # typically affects a narrow band, we inpaint a cropped region and paste back.
        bbox = _mask_bbox(mask_gray)
        if bbox is None:
            # Nothing to inpaint
            return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in ori_frames_rgb]

        margin = int(getattr(config, "E2FGVI_CROP_MARGIN", 96))
        max_side = int(getattr(config, "E2FGVI_MAX_CROP_SIDE", 960))
        margin = max(0, min(margin, 400))
        max_side = max(256, min(max_side, 1920))

        y0, y1, x0, x1 = bbox
        y0 = _clamp(y0 - margin, 0, orig_h)
        y1 = _clamp(y1 + margin, 0, orig_h)
        x0 = _clamp(x0 - margin, 0, orig_w)
        x1 = _clamp(x1 + margin, 0, orig_w)
        if y1 <= y0 or x1 <= x0:
            return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in ori_frames_rgb]

        crop_h = int(y1 - y0)
        crop_w = int(x1 - x0)

        # Crop original frames (RGB) and mask (gray) at full-res for later paste-back
        ori_crop_rgb = [f[y0:y1, x0:x1, :] for f in ori_frames_rgb]
        mask_crop_full = mask_gray[y0:y1, x0:x1]

        # Downscale crop if needed (keeps quality acceptable for subtitle band)
        scale = 1.0
        if max(crop_h, crop_w) > max_side:
            scale = float(max_side) / float(max(crop_h, crop_w))
        proc_w = max(16, int(round(crop_w * scale)))
        proc_h = max(16, int(round(crop_h * scale)))
        if (proc_w, proc_h) != (crop_w, crop_h):
            pil_frames = [Image.fromarray(cv2.resize(f, (proc_w, proc_h), interpolation=cv2.INTER_AREA)) for f in ori_crop_rgb]
            mask_gray = cv2.resize(mask_crop_full, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        else:
            pil_frames = [Image.fromarray(f) for f in ori_crop_rgb]
            mask_gray = mask_crop_full

        # ── 2. Build tensors [1, T, C, H, W] ──────────────────────────────────
        _to_tensor = transforms.ToTensor()   # → float32 in [0, 1]

        frames_t = torch.stack([_to_tensor(f) for f in pil_frames], dim=0)   # [T, 3, H, W]
        frames_t = frames_t.unsqueeze(0)                                       # [1, T, 3, H, W]
        # E2FGVI expects input in [-1, 1]; model internally converts to [0,1] for SPyNet
        frames_t = frames_t * 2 - 1

        mask_t = _to_tensor(Image.fromarray(mask_gray))                        # [1, H, W]
        mask_t = (mask_t > 0.5).float()                                        # binary
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)                              # [1, 1, 1, H, W]
        # expand across time; spatial size is now the (possibly downscaled) crop
        mask_t = mask_t.expand(1, video_length, 1, mask_t.shape[-2], mask_t.shape[-1]).clone()    # [1, T, 1, H, W]

        # Pad H to multiple of 60, W to multiple of 108 (matches official test.py)
        frames_t, (pad_h, pad_w) = _pad_to_model_size(frames_t)
        mask_t,   _              = _pad_to_model_size(mask_t)

        use_fp16 = bool(getattr(config, "E2FGVI_USE_FP16", True)) and (self.device.type == "cuda")
        if use_fp16:
            if not self._model_is_half:
                self.model = self.model.half()
                self._model_is_half = True
            frames_t = frames_t.half()
            mask_t = mask_t.half()
        elif self._model_is_half:
            self.model = self.model.float()
            self._model_is_half = False

        frames_t = frames_t.to(self.device, non_blocking=True)
        mask_t   = mask_t.to(self.device, non_blocking=True)

        # ── 3. Sliding-window temporal inference ───────────────────────────────
        comp_frames    = [None] * video_length
        neighbor_stride = self.neighbor_length // 2

        with torch.no_grad():
            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = [
                    i for i in range(max(0, f - neighbor_stride),
                                     min(video_length, f + neighbor_stride + 1))
                ]
                ref_ids  = _get_ref_index(f, neighbor_ids, video_length, self.ref_stride)
                all_ids  = neighbor_ids + ref_ids

                # Select frames and masks for this window
                sel_frames = frames_t[:, all_ids]   # [1, T_sel, 3, H, W]
                sel_masks  = mask_t[:,   all_ids]   # [1, T_sel, 1, H, W]

                # Zero out the inpaint region — model sees only valid pixels
                masked_frames = sel_frames * (1.0 - sel_masks)

                # Forward pass — model expects [B, T, C, H, W] (5D, B=1)
                # Returns [B*T_all, 3, H, W] with values in [-1, 1] (tanh output)
                if use_fp16 and torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        pred_img, _ = self.model(masked_frames, len(neighbor_ids))
                else:
                    pred_img, _ = self.model(masked_frames, len(neighbor_ids))
                # Convert [-1, 1] → [0, 1], crop padding, clamp
                pred_img = (pred_img + 1) / 2
                pred_img = torch.clamp(pred_img, 0.0, 1.0)

                # Back to numpy uint8
                pred_np    = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255.0
                bin_masks  = mask_t[0, neighbor_ids].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

                for i, idx in enumerate(neighbor_ids):
                    pred_crop = pred_np[i].astype(np.uint8)                          # (Hc, Wc, 3) padded
                    mask_crop = bin_masks[i]                                          # (Hc, Wc, 1) padded

                    # Remove padding (back to processed crop size)
                    ph = pred_crop.shape[0] - pad_h
                    pw = pred_crop.shape[1] - pad_w
                    if ph > 0 and pw > 0:
                        pred_crop = pred_crop[:ph, :pw, :]
                        mask_crop = mask_crop[:ph, :pw, :]

                    # Resize back to full-res crop if we downscaled
                    if (pred_crop.shape[1], pred_crop.shape[0]) != (crop_w, crop_h):
                        pred_crop = cv2.resize(pred_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                        mask_crop = cv2.resize(mask_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                        if mask_crop.ndim == 2:
                            mask_crop = mask_crop[:, :, None]

                    # Composite inside crop region (RGB)
                    src_crop = ori_crop_rgb[idx]

                    # Composite: inpainted pixels inside mask, original outside
                    composited = (pred_crop * mask_crop + src_crop * (1 - mask_crop)).astype(np.uint8)

                    if comp_frames[idx] is None:
                        comp_frames[idx] = composited
                    else:
                        # Blend overlapping window predictions (same as ProPainter)
                        comp_frames[idx] = (
                            comp_frames[idx].astype(np.float32) * 0.5 +
                            composited.astype(np.float32) * 0.5
                        ).astype(np.uint8)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ── 4. Paste crop back into full frames, RGB → BGR ────────────────────
        out_full = []
        for i in range(video_length):
            base = ori_frames_rgb[i].copy()
            # Use full-res crop mask for paste-back
            m = (mask_crop_full > 0).astype(np.uint8)[:, :, None]
            filled = comp_frames[i]
            # Blend filled crop into base using the mask (no feather here; main pipeline blends later)
            base[y0:y1, x0:x1, :] = (filled * m + base[y0:y1, x0:x1, :] * (1 - m)).astype(np.uint8)
            out_full.append(cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
        return out_full
