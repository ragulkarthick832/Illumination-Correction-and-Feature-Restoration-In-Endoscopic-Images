import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from itertools import chain

# adjust imports to your filenames
from dataloader import Endo3InputDataset     # your dataset file
from models.generator import GeneratorUNet   # fixed generator
from models.discriminator import NLayerDiscriminator

# ---------------------------
# Config / hyperparams
# ---------------------------
root = "D:/Final_year_Endoscopy/Dataset"  # update if needed
paired_root = os.path.join(root)  # your existing dataset root that dataloader expects
out_dir = "checkpoints"
sample_dir = "samples"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(sample_dir, exist_ok=True)

img_size = 256
batch_size = 4         # try 4; increase if you have room
num_workers = 0
lr = 2e-4
beta1, beta2 = 0.5, 0.999
num_epochs = 200
lambda_l1 = 100.0      # weight for L1 reconstruction loss (common for paired pix2pix-like)
save_every = 5         # save checkpoint every N epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Prepare dataset + dataloader
# ---------------------------
train_ds = Endo3InputDataset(root=paired_root, split="train", img_size=img_size)
val_ds   = Endo3InputDataset(root=paired_root, split="validation", img_size=img_size)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# ---------------------------
# Models
# ---------------------------
G = GeneratorUNet(input_nc=9, output_nc=3).to(device)
D = NLayerDiscriminator(input_nc=3).to(device)

# initialize weights
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G.apply(init_weights)
D.apply(init_weights)

# ---------------------------
# Losses + optimizers
# ---------------------------
criterion_GAN = nn.BCEWithLogitsLoss()  # discriminator outputs raw logits
criterion_L1 = nn.L1Loss()

optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

# Learning rate schedulers (optional)
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda step: 1.0)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda step: 1.0)

# ---------------------------
# Helper utilities
# ---------------------------
def tensor2img(t):
    # t in [0,1] expected
    return t.clamp(0,1)

def save_sample(inp_tensor, out_tensor, gt_tensor, epoch, step):
    # inp_tensor shape (B,9,H,W)
    # out_tensor, gt_tensor shape (B,3,H,W)
    b = min(4, out_tensor.size(0))
    # take first b samples
    outs = out_tensor[:b]
    gts  = gt_tensor[:b]
    # create a grid: [input fused image | output | gt]
    # make a fused RGB input for visualization: average of 3 inputs
    inp_rgb = inp_tensor.view(inp_tensor.size(0), 3, 3, inp_tensor.size(2), inp_tensor.size(3)).mean(dim=2)
    grid = torch.cat([inp_rgb[:b], outs, gts], dim=0)
    save_image(tensor2img(grid), os.path.join(sample_dir, f"epoch{epoch:03d}_step{step:04d}.png"), nrow=b)

# ---------------------------
# Training loop
# ---------------------------
def train():
    fixed_samples = None
    for epoch in range(1, num_epochs+1):
        G.train(); D.train()
        epoch_start = time.time()
        running_D_loss = 0.0
        running_G_loss = 0.0

        for step, (inp, gt) in enumerate(train_loader, start=1):
            inp = inp.to(device)   # (B,9,H,W)
            gt  = gt.to(device)    # (B,3,H,W)
            bs = inp.size(0)

            # ----------------------
            # Update D: maximize log(D(real)) + log(1 - D(fake))
            # ----------------------
            optimizer_D.zero_grad()

            # real
            pred_real = D(gt)              # (B,1,h,w)
            real_label = torch.ones_like(pred_real, device=device)
            loss_D_real = criterion_GAN(pred_real, real_label)

            # fake
            fake = G(inp).detach()
            pred_fake = D(fake)
            fake_label = torch.zeros_like(pred_fake, device=device)
            loss_D_fake = criterion_GAN(pred_fake, fake_label)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ----------------------
            # Update G: adversarial + L1
            # ----------------------
            optimizer_G.zero_grad()
            fake_for_G = G(inp)
            pred_fake_for_G = D(fake_for_G)
            adv_label = torch.ones_like(pred_fake_for_G, device=device)  # want D(fake) -> real
            loss_G_GAN = criterion_GAN(pred_fake_for_G, adv_label)
            loss_G_L1 = criterion_L1(fake_for_G, gt) * lambda_l1

            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            running_D_loss += loss_D.item()
            running_G_loss += loss_G.item()

            # save a small visual sample occasionally
            if step % 200 == 0:
                save_sample(inp.cpu(), fake_for_G.detach().cpu(), gt.cpu(), epoch, step)

        # epoch end
        epoch_time = time.time() - epoch_start
        avg_D = running_D_loss / len(train_loader)
        avg_G = running_G_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs}  D_loss={avg_D:.4f}  G_loss={avg_G:.4f}  time={epoch_time:.1f}s")

        # save checkpoint
        if epoch % save_every == 0 or epoch == 1:
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict()
            }, os.path.join(out_dir, f"checkpoint_epoch{epoch:03d}.pth"))

        # scheduler step (if using)
        scheduler_G.step()
        scheduler_D.step()

        # small validation save
        if epoch % save_every == 0:
            G.eval()
            with torch.no_grad():
                # save first N val samples
                for i, (inp_v, gt_v) in enumerate(val_loader):
                    if i >= 4: break
                    inp_v = inp_v.to(device)
                    pred_v = G(inp_v)
                    # make fused input RGB for visualization (average of 3)
                    inp_rgb_v = inp_v.view(inp_v.size(0), 3, 3, inp_v.size(2), inp_v.size(3)).mean(dim=2)
                    grid = torch.cat([inp_rgb_v.cpu(), pred_v.cpu(), gt_v], dim=0)
                    save_image(tensor2img(grid), os.path.join(sample_dir, f"val_epoch{epoch:03d}_{i}.png"), nrow=1)
if __name__ == "__main__":
    train()