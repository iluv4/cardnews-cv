"""Quick reality-check: run the pretrained DS-GAN (PosterLayout) on a few REAL
Korean card-news images and draw the predicted layout boxes.

- Saliency is computed on the fly with a lightweight spectral-residual method
  (Stage 3 of the project later swaps this for a real U^2-Net / BASNet).
- Caveat: these are *finished* cards (text already present), not clean inpainted
  backgrounds, so this only checks that the model runs and predicts plausible
  regions — not final layout quality.

Run from the repo root:  .venv\Scripts\python.exe test_real.py
"""
import os
import glob
from collections import OrderedDict

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw

from model import generator

SRC_DIR = r"C:\Users\Administrator\Downloads\carnews-insta\public\saved-refs"
OUT_DIR = "output/real_test"
CKPT = "output/DS-GAN-Epoch300.pth"
N = 6
MAX_ELEM = 32
W, H = 513, 750

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)


def spectral_saliency(pil_rgb):
    """Spectral-residual saliency (Hou & Zhang). Returns an 'L' PIL map, WxH."""
    gray = cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(gray, (64, 64))
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    spectral_residual = log_amp - cv2.blur(log_amp, (3, 3))
    recon = np.fft.ifft2(np.exp(spectral_residual + 1j * phase))
    sal = np.abs(recon) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), 2.5)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal = cv2.resize(sal, (W, H))
    return Image.fromarray((sal * 255).astype(np.uint8), "L")


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    return torch.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], dim=-1)


def random_init(batch, max_elem):
    coef = [0.1, 0.8, 1, 1]
    cls_1 = torch.tensor(np.random.choice(4, size=(batch, max_elem, 1), p=np.array(coef) / sum(coef))).long()
    cls = torch.zeros((batch, max_elem, 4))
    cls.scatter_(-1, cls_1, 1)
    box_xyxy = torch.normal(0.5, 0.15, size=(batch, max_elem, 1, 4))
    x0, y0, x1, y1 = box_xyxy.unbind(-1)
    box = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)
    return torch.concat([cls.unsqueeze(2), box], dim=2)


def main():
    paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.*")))[:N]
    paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"device={device}  images={len(paths)}")
    if not paths:
        raise SystemExit(f"No images found in {SRC_DIR}")

    tf = transforms.Compose([transforms.Resize([350, 240]), transforms.ToTensor()])

    inputs, originals = [], []
    for p in paths:
        rgb = Image.open(p).convert("RGB").resize((W, H))
        sal = spectral_saliency(rgb)
        originals.append(rgb)
        cc = torch.concat([tf(rgb), tf(sal)])  # 4 channels: RGB + saliency
        inputs.append(cc)
    batch = torch.stack(inputs).to(device)

    # load pretrained generator (strip DataParallel 'module.' prefix)
    G = generator({"backbone": "resnet50", "in_channels": 8, "out_channels": 32,
                   "hidden_size": MAX_ELEM * 8, "num_layers": 4, "output_size": 8,
                   "max_elem": MAX_ELEM})
    ckpt = torch.load(CKPT, map_location="cpu")
    sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in ckpt.items())
    G.load_state_dict(sd)
    G = G.to(device).eval()

    init_layout = random_init(len(paths), MAX_ELEM).to(device)
    with torch.no_grad():
        cls, box = G(batch, init_layout)

    cls = torch.argmax(cls.cpu(), dim=-1)          # 0=pad,1=text,2=logo,3=underlay
    box = box_cxcywh_to_xyxy(box.cpu())
    box[:, :, ::2] *= W
    box[:, :, 1::2] *= H

    colors = {1: (0, 200, 0), 2: (220, 0, 0), 3: (255, 140, 0)}
    names = {1: "text", 2: "logo", 3: "underlay"}
    for i, (img, c_i, b_i) in enumerate(zip(originals, cls, box)):
        drawn = img.copy()
        d = ImageDraw.Draw(drawn)
        n_elem = 0
        for c, b in zip(c_i.tolist(), b_i.tolist()):
            if c == 0:
                continue
            n_elem += 1
            d.rectangle(b, outline=colors.get(c, (0, 0, 0)), width=5)
        out = os.path.join(OUT_DIR, f"layout_{i}_{os.path.basename(paths[i])}.png")
        drawn.save(out)
        kinds = [names[c] for c in c_i.tolist() if c]
        print(f"[{i}] {os.path.basename(paths[i])}: {n_elem} elements -> {kinds}")
    print(f"\nSaved overlays to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
