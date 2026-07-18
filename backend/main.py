"""
SegNCA API — serves live inference from two real trained checkpoints
(SegNCA: 8,320 params · TinyUNet: 483,153 params) trained on real clinical
hippocampus MRI data (Medical Segmentation Decathlon, Task04).

Run: uvicorn main:app --reload --port 8000
"""
import base64
import io
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from models import SegNCA, TinyUNet, dice_score, iou_score

app = FastAPI(title="SegNCA API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
CKPT_DIR = "checkpoints"

# --- load everything once at startup ---
images = torch.tensor(np.load(f"{DATA_DIR}/images.npy"))
labels = torch.tensor(np.load(f"{DATA_DIR}/labels.npy"))
val_idx = np.load(f"{DATA_DIR}/val_idx.npy")

nca = SegNCA()
nca.load_state_dict(torch.load(f"{CKPT_DIR}/segnca_best.pt", map_location="cpu"))
nca.eval()

unet = TinyUNet()
unet.load_state_dict(torch.load(f"{CKPT_DIR}/tinyunet_best.pt", map_location="cpu"))
unet.eval()

N_NCA_PARAMS = sum(p.numel() for p in nca.parameters())
N_UNET_PARAMS = sum(p.numel() for p in unet.parameters())


def tensor_to_b64_png(arr: np.ndarray) -> str:
    """arr: (H,W) or (H,W,3) float in [0,1] -> base64 PNG string."""
    arr = np.clip(arr, 0, 1)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay(base_img: np.ndarray, mask: np.ndarray, color, alpha=0.45) -> np.ndarray:
    base = np.stack([base_img, base_img, base_img], axis=-1)
    m = mask > 0.5
    out = base.copy()
    for c in range(3):
        out[..., c] = np.where(m, base[..., c] * (1 - alpha) + color[c] * alpha, base[..., c])
    return out


class CorruptRequest(BaseModel):
    sample_idx: int = 0
    noise_sigma: float = 0.0
    shift_px: int = 0
    nca_runs: int = 5  # how many stochastic NCA forward passes to average


@app.get("/api/health")
def health():
    return {"status": "ok", "nca_params": N_NCA_PARAMS, "unet_params": N_UNET_PARAMS}


@app.get("/api/metrics")
def get_metrics():
    import json
    with open(f"{DATA_DIR}/final_metrics.json") as f:
        return json.load(f)


@app.get("/api/robustness")
def get_robustness():
    import json
    with open(f"{DATA_DIR}/robustness.json") as f:
        return json.load(f)


@app.get("/api/samples")
def list_samples():
    """Thumbnails for every validation slice, so the frontend can offer a picker."""
    out = []
    for i, idx in enumerate(val_idx):
        img = images[idx].numpy()
        out.append({"sample_idx": i, "thumbnail": tensor_to_b64_png(img)})
    return {"samples": out}


@app.post("/api/predict")
def predict(req: CorruptRequest):
    if not (0 <= req.sample_idx < len(val_idx)):
        raise HTTPException(400, f"sample_idx must be in [0, {len(val_idx)})")

    x = images[val_idx[req.sample_idx]].clone().unsqueeze(0)  # (1,H,W)
    y = labels[val_idx[req.sample_idx]].clone().unsqueeze(0)

    # apply live corruption
    if req.noise_sigma > 0:
        torch.manual_seed(0)
        x = (x + torch.randn_like(x) * req.noise_sigma).clamp(0, 1)
    if req.shift_px != 0:
        s = req.shift_px
        x = torch.roll(x, shifts=(s, s), dims=(1, 2))
        y = torch.roll(y, shifts=(s, s), dims=(1, 2))
        if s > 0:
            x[:, :s, :] = 0
            y[:, :s, :] = 0
            x[:, :, :s] = 0
            y[:, :, :s] = 0

    with torch.no_grad():
        t0 = time.time()
        nca_dices, nca_probs_runs = [], []
        for _ in range(max(1, req.nca_runs)):
            logits = nca(x.unsqueeze(-1), steps=28, fire_rate=0.5)
            nca_probs_runs.append(torch.sigmoid(logits))
            nca_dices.append(dice_score(logits, y))
        nca_latency_ms = (time.time() - t0) / max(1, req.nca_runs) * 1000
        nca_probs = torch.stack(nca_probs_runs).mean(0)[0].numpy()
        nca_dice_mean = float(np.mean(nca_dices))
        nca_dice_std = float(np.std(nca_dices))

        t0 = time.time()
        unet_logits = unet(x.unsqueeze(1))
        unet_latency_ms = (time.time() - t0) * 1000
        unet_probs = torch.sigmoid(unet_logits)[0].numpy()
        unet_dice = dice_score(unet_logits, y)
        unet_iou = iou_score(unet_logits, y)
        nca_iou = iou_score(torch.log(torch.clamp(torch.tensor(nca_probs), 1e-6, 1 - 1e-6) /
                                       torch.clamp(1 - torch.tensor(nca_probs), 1e-6, 1 - 1e-6)).unsqueeze(0), y)

    img_np = x[0].numpy()
    gt_np = y[0].numpy()
    img_u8 = np.clip(img_np, 0, 1)

    return {
        "input_image": tensor_to_b64_png(img_u8),
        "ground_truth": tensor_to_b64_png(overlay(img_u8, gt_np, (80, 220, 80))),
        "nca_prediction": tensor_to_b64_png(overlay(img_u8, nca_probs, (255, 70, 70))),
        "unet_prediction": tensor_to_b64_png(overlay(img_u8, unet_probs, (70, 140, 255))),
        "nca": {"dice": nca_dice_mean, "dice_std": nca_dice_std, "iou": nca_iou, "latency_ms": nca_latency_ms, "params": N_NCA_PARAMS},
        "unet": {"dice": unet_dice, "iou": unet_iou, "latency_ms": unet_latency_ms, "params": N_UNET_PARAMS},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
