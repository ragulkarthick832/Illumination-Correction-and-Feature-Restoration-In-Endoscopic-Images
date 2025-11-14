import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import csv

from models.generator import GeneratorUNet
from math import log10
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------
# PATHS (CHANGE ONLY THIS ROOT)
# -----------------------------------------------------------
DATASET_ROOT = "D:/Final_year_Endoscopy/Dataset"

linear_dir  = os.path.join(DATASET_ROOT, "linear", "test")
radial_dir  = os.path.join(DATASET_ROOT, "radial", "test")
corner_dir  = os.path.join(DATASET_ROOT, "corner", "test")
gt_dir      = os.path.join(DATASET_ROOT, "gt", "test")
out_dir     = "pred_test_out"
compare_dir = "compare"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(compare_dir, exist_ok=True)

# -----------------------------------------------------------
# LOAD GENERATOR
# -----------------------------------------------------------
G = GeneratorUNet(9, 3).to(device)

ckpt_path = "checkpoints/checkpoint_epoch200.pth"
print("Loading checkpoint:", ckpt_path)

ckpt = torch.load(ckpt_path, map_location=device)
G.load_state_dict(ckpt["G_state_dict"])
G.eval()

# -----------------------------------------------------------
# TRANSFORMS
# -----------------------------------------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])


# -----------------------------------------------------------
# METRICS
# -----------------------------------------------------------
def mse(pred, gt):
    return torch.mean((pred - gt) ** 2).item()

def psnr(pred, gt):
    mse_v = mse(pred, gt)
    if mse_v == 0:
        return 100
    return 20 * log10(1.0 / (mse_v ** 0.5))

def ssim(pred, gt):
    # simple PyTorch SSIM approximate using gaussian blur
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 0)
    mu_y = F.avg_pool2d(gt, 3, 1, 0)

    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 0) - mu_x * mu_x
    sigma_y = F.avg_pool2d(gt * gt, 3, 1, 0) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * gt, 3, 1, 0) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))

    return ssim_map.mean().item()


# -----------------------------------------------------------
# LOAD TRIPLET
# -----------------------------------------------------------
def load_triplet(id):
    A = transform(Image.open(os.path.join(linear_dir,  f"{id}_linear.jpg")).convert("RGB"))
    B = transform(Image.open(os.path.join(radial_dir,  f"{id}_radial.jpg")).convert("RGB"))
    C = transform(Image.open(os.path.join(corner_dir, f"{id}_corner.jpg")).convert("RGB"))
    x = torch.cat([A, B, C], dim=0)
    return x.unsqueeze(0).to(device)   # (1,9,256,256)

def load_gt(id):
    return transform(Image.open(os.path.join(gt_dir, f"{id}.jpg")).convert("RGB")).unsqueeze(0).to(device)


# -----------------------------------------------------------
# RUN PREDICTION FOR ALL TEST IMAGES
# -----------------------------------------------------------
files = sorted(os.listdir(linear_dir))
ids = [f.replace("_linear.jpg", "") for f in files if f.endswith("_linear.jpg")]

print(f"Found {len(ids)} test images")

# CSV metrics file
csv_path = "metrics.csv"
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["ID", "PSNR", "SSIM", "MSE"])

with torch.no_grad():
    for idx, id in enumerate(ids, start=1):
        
        # ---- Load Input & GT ----
        inp = load_triplet(id)     # (1,9,256,256)
        gt = load_gt(id)           # (1,3,256,256)

        # ---- Predict ----
        pred = G(inp)              # (1,3,256,256)

        # Save output
        save_image(pred, os.path.join(out_dir, f"{id}.jpg"))

        # ---- Compute Metrics ----
        P = pred.cpu()
        Gt = gt.cpu()

        v_psnr = psnr(P, Gt)
        v_ssim = ssim(P, Gt)
        v_mse  = mse(P, Gt)

        writer.writerow([id, v_psnr, v_ssim, v_mse])

        # ---- Save side-by-side ----
        inp_rgb = inp.view(1, 3, 3, 256, 256).mean(dim=2)   # 3-channel fused version

        comp = torch.cat([inp_rgb.cpu(), pred.cpu(), gt.cpu()], dim=0)
        save_image(comp, os.path.join(compare_dir, f"{id}_compare.jpg"), nrow=3)

        if idx % 50 == 0:
            print(f"Processed {idx}/{len(ids)}")

csv_file.close()
print("\nDONE!")
print("Predictions saved in:", out_dir)
print("Comparison images saved in:", compare_dir)
print("Metrics saved in:", csv_path)
