import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class Endo3InputDataset(Dataset):
    def __init__(self, root, split='train', img_size=256):
        """
        root: path to Endo4IE/paired
        split: 'train', 'val', or 'test'
        """
        self.root = root
        self.split = split

        # Directories
        self.dir_A = os.path.join(root, "linear", split)
        self.dir_B = os.path.join(root, "radial", split)
        self.dir_C = os.path.join(root, "corner", split)
        self.dir_GT = os.path.join(root, "gt", split)

        # Sorted file list by ID prefix
        gt_files = sorted(os.listdir(self.dir_GT))

        # Extract ids: 00001.png → 00001
        self.ids = [f.split(".")[0] for f in gt_files]

        # Resize + convert to tensor
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()         # converts to [0,1]
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        # File names
        file_A = id + "_linear.jpg"
        file_B = id + "_radial.jpg"
        file_C = id + "_corner.jpg"
        file_GT = id + ".jpg"

        # Load images
        img_A = Image.open(os.path.join(self.dir_A, file_A)).convert("RGB")
        img_B = Image.open(os.path.join(self.dir_B, file_B)).convert("RGB")
        img_C = Image.open(os.path.join(self.dir_C, file_C)).convert("RGB")
        img_GT = Image.open(os.path.join(self.dir_GT, file_GT)).convert("RGB")

        # Apply transform
        img_A = self.transform(img_A)   # shape (3,H,W)
        img_B = self.transform(img_B)
        img_C = self.transform(img_C)
        img_GT = self.transform(img_GT)

        # Stack 3 images → 9 channels
        inp = torch.cat([img_A, img_B, img_C], dim=0)  # (9,H,W)

        return inp, img_GT
