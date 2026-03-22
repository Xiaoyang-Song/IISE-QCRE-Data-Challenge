import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
import sys
sys.path.append("../")
from utils import *


class LabeledDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None, cropped=False):
        """
        root_dir: Data/train_labeled
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "Labeled Images")
        self.csv_path = os.path.join(root_dir, "train_labels.csv")
        self.cropped = cropped

        # Load CSV
        self.df = pd.read_csv(self.csv_path)

        # Clean potential issues (important)
        self.df = self.df.dropna(subset=["Image_id"])
        self.df = self.df.reset_index(drop=True)

        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Precompute paths + labels
        self.samples = []

        for _, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row["Image_id"])

            if not os.path.exists(img_path):
                continue  # skip missing files safely

            label = torch.tensor([
                # row["Defect"],
                row["DT1_MP"],
                row["DT2_TP"],
                row["DT3_OOB"]
            ], dtype=torch.float32)

            self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Transform
        if self.cropped:
            # PIL RGB -> numpy RGB
            img_np = np.array(image)
            # For anchor detection, use grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # Detect anchors on grayscale
            anchors = fire_the_hole(gray)

            # Crop from the original RGB image
            crops = image2crops_with_anchors(img_np, anchors)
            # Make sure crops is a list
            if crops is None:
                crops = []
            # Convert numpy crops to PIL
            pil_crops = []
            for c in crops[:4]:
                if c is None or c.size == 0:
                    continue
                pil_crops.append(Image.fromarray(c))

            # Pad to exactly 4 crops with black background
            while len(pil_crops) < 4:
                pil_crops.append(Image.new("RGB", image.size, (0, 0, 0)))

            # Transform each crop: [3, 224, 224]
            crop_tensors = [self.transform(c) for c in pil_crops]

            # Stack -> [4, 3, 224, 224]
            crop_tensors = torch.stack(crop_tensors, dim=0)

            return crop_tensors, label
        else:
            image = self.transform(image)
            return image, label


if __name__ == "__main__":
    
    dataset = LabeledDefectDataset("Data/train_labeled", cropped=True)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    for batch, y in loader:
        print(batch.shape, y.shape)