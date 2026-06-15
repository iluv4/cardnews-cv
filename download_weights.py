"""Re-download the public pretrained weights (so they don't need to live in git).
Run after creating the venv. Places weights where the PosterLayout code expects.
"""
import os
import urllib.request

TARGETS = {
    "../PosterLayout-CVPR2023/model_weight/resnet18-5c106cde.pth":
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "../PosterLayout-CVPR2023/model_weight/resnet50_a1_0-14fe96d1.pth":
        "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
}
# DS-GAN checkpoint lives in a public Google Drive folder -> use gdown.
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1UYJ34BhqgYztfh5n5A4GU4nqgboPtoWS"
DSGAN_DEST = "../PosterLayout-CVPR2023/output"

for dest, url in TARGETS.items():
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"exists: {dest}")
        continue
    print(f"downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  ok ({os.path.getsize(dest)//1024//1024} MB)")

print("\nFor DS-GAN-Epoch300.pth run:")
print(f"  gdown --folder {GDRIVE_FOLDER} -O {DSGAN_DEST}")
print("  (keep DS-GAN-Epoch300.pth in output/, the two resnet .pth go in model_weight/)")
