import os
import csv
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    # data
    batch_size: int = 128
    num_workers: int = 8
    image_size: int = 224

    # optimization
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.2

    # model
    backbone_name: str = "resnet18"   # "resnet18", "resnet50"
    pretrained: bool = True
    projection_dim: int = 128
    hidden_dim: int = 512

    # system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    seed: int = 42

    # save / log
    save_dir: str = "./simclr_runs"
    run_name: str = "simclr_resnet18"
    log_interval: int = 20
    save_every: int = 5
    resume_path: str = ""   # path to checkpoint if resuming


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_config(cfg: TrainConfig, out_dir: str):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "config.txt"), "w") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")


def write_csv_log_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "lr", "epoch_time_sec"
            ])


def append_csv_log(csv_path: str, epoch: int, train_loss: float, lr: float, epoch_time_sec: float):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, lr, epoch_time_sec])


# ============================================================
# SimCLR Augmentations
# ============================================================

class SimCLRTransform:
    """
    Produce two augmented views of the same image.
    Input can be PIL image or tensor.
    """
    def __init__(self, image_size=224):
        color_jitter = transforms.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )

        self.base_transform = transforms.Compose([
            transforms.ToPILImage() if not isinstance(image_size, tuple) else transforms.Lambda(lambda x: x),
        ])

        self.transform = transforms.Compose([
            transforms.ToPILImage() if True else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, x):
        # x expected as tensor [C,H,W] or PIL image
        if isinstance(x, torch.Tensor):
            if x.dtype != torch.uint8:
                # assume float tensor; clamp to valid range for PIL conversion
                x = x.detach().cpu()
                if x.max() <= 1.0:
                    x = (x * 255.0).clamp(0, 255).byte()
                else:
                    x = x.clamp(0, 255).byte()
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


# ============================================================
# Dataset Wrapper
# ============================================================

class SimCLRDatasetWrapper(Dataset):
    """
    Wrap an existing dataset so it returns two augmented views.
    Supports:
      - dataset[idx] -> image
      - dataset[idx] -> (image, label)
      - dataset[idx] -> {"image": image, ...}
    """
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def _extract_image(self, sample):
        if isinstance(sample, torch.Tensor):
            return sample
        if isinstance(sample, (tuple, list)):
            return sample[0]
        if isinstance(sample, dict):
            if "image" in sample:
                return sample["image"]
            raise ValueError("Dict sample found but no 'image' key exists.")
        raise ValueError(f"Unsupported sample type: {type(sample)}")

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image = self._extract_image(sample)
        x1, x2 = self.transform(image)
        return x1, x2


# ============================================================
# Model
# ============================================================

class ProjectionHead(nn.Module):
    """
    Standard SimCLR MLP projection head:
    encoder_out -> hidden -> projection_dim
    """
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        pretrained=True,
        projection_dim=128,
        hidden_dim=512
    ):
        super().__init__()

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            feat_dim = backbone.fc.in_features
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            feat_dim = backbone.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # remove classification head
        backbone.fc = nn.Identity()

        self.encoder = backbone
        self.projector = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=projection_dim
        )

    def forward(self, x):
        h = self.encoder(x)            # representation
        z = self.projector(h)          # projection
        z = F.normalize(z, dim=1)      # important for contrastive learning
        return h, z


# ============================================================
# NT-Xent Loss
# ============================================================

class NTXentLoss(nn.Module):
    """
    Normalized temperature-scaled cross entropy loss used in SimCLR.
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        z1, z2: [B, D], already normalized
        """
        batch_size = z1.size(0)
        device = z1.device

        z = torch.cat([z1, z2], dim=0)   # [2B, D]

        # similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        # mask self-similarity
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)

        # positive pairs:
        # for i in [0..B-1], positive is i+B
        # for i in [B..2B-1], positive is i-B
        targets = torch.arange(batch_size, 2 * batch_size, device=device)
        targets = torch.cat([targets, torch.arange(0, batch_size, device=device)], dim=0)

        loss = F.cross_entropy(sim, targets)
        return loss


# ============================================================
# Scheduler
# ============================================================

def build_cosine_scheduler(optimizer, total_epochs, warmup_epochs=10, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / optimizer.defaults["lr"], cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_loss):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_loss": best_loss,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = ckpt.get("epoch", -1)
    best_loss = ckpt.get("best_loss", float("inf"))
    return epoch, best_loss


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, cfg):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for step, (x1, x2) in enumerate(loader):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.startswith("cuda"))):
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)

        if scaler is not None and cfg.use_amp and device.startswith("cuda"):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if (step + 1) % cfg.log_interval == 0:
            avg_so_far = running_loss / (step + 1)
            print(
                f"Epoch [{epoch}] Step [{step+1}/{len(loader)}] "
                f"Loss: {avg_so_far:.4f}"
            )

    epoch_loss = running_loss / max(1, len(loader))
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_time


# ============================================================
# Main Train Function
# ============================================================

def train_simclr(base_dataset, cfg: TrainConfig):
    set_seed(cfg.seed)

    run_dir = os.path.join(cfg.save_dir, cfg.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    save_config(cfg, run_dir)

    csv_log_path = os.path.join(run_dir, "train_log.csv")
    write_csv_log_header(csv_log_path)

    # dataset / loader
    simclr_transform = SimCLRTransform(image_size=cfg.image_size)
    train_dataset = SimCLRDatasetWrapper(base_dataset, simclr_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,   # important for contrastive learning
        persistent_workers=(cfg.num_workers > 0),
    )

    # model
    model = SimCLRModel(
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim
    ).to(cfg.device)

    criterion = NTXentLoss(temperature=cfg.temperature)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scheduler = build_cosine_scheduler(optimizer, total_epochs=cfg.epochs, warmup_epochs=10)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device.startswith("cuda")))

    start_epoch = 0
    best_loss = float("inf")

    # resume
    if cfg.resume_path and os.path.isfile(cfg.resume_path):
        print(f"Resuming from checkpoint: {cfg.resume_path}")
        last_epoch, best_loss = load_checkpoint(
            cfg.resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=cfg.device
        )
        start_epoch = last_epoch + 1
        print(f"Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")

    # training loop
    for epoch in range(start_epoch, cfg.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print("=" * 80)
        print(f"Epoch {epoch}/{cfg.epochs - 1} | LR: {current_lr:.8f}")

        train_loss, epoch_time = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=cfg.device,
            epoch=epoch,
            cfg=cfg
        )

        scheduler.step()

        print(
            f"Epoch {epoch} finished | "
            f"Train Loss: {train_loss:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        append_csv_log(
            csv_log_path,
            epoch=epoch,
            train_loss=train_loss,
            lr=current_lr,
            epoch_time_sec=epoch_time
        )

        # save latest
        latest_path = os.path.join(ckpt_dir, "latest.pt")
        save_checkpoint(
            latest_path, model, optimizer, scheduler, scaler, epoch, best_loss
        )

        # save best
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler, epoch, best_loss
            )
            print(f"Saved new best checkpoint to: {best_path}")

        # periodic save
        if (epoch + 1) % cfg.save_every == 0:
            epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                epoch_path, model, optimizer, scheduler, scaler, epoch, best_loss
            )
            print(f"Saved checkpoint to: {epoch_path}")

    print("=" * 80)
    print("Training finished.")
    print(f"Best training loss: {best_loss:.4f}")
    print(f"Run directory: {run_dir}")

    return model


# ============================================================
# Feature Extraction Helper
# ============================================================

@torch.no_grad()
def extract_features(model, loader, device="cuda"):
    """
    Extract encoder features h, not projection z.
    """
    model.eval()
    all_features = []

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device, non_blocking=True)
        h, _ = model(x)
        all_features.append(h.cpu())

    return torch.cat(all_features, dim=0)


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Replace this with your actual dataset
    # Example:
    # from my_dataset import MyImageDataset
    # base_dataset = MyImageDataset(...)
    # --------------------------------------------------------

    class DummyDataset(Dataset):
        def __init__(self, n=1000):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            # fake image in [0,1]
            img = torch.rand(3, 224, 224)
            return img

    base_dataset = DummyDataset(n=2000)

    cfg = TrainConfig(
        batch_size=64,
        epochs=50,
        lr=3e-4,
        backbone_name="resnet18",
        pretrained=True,
        projection_dim=128,
        hidden_dim=512,
        save_dir="./simclr_runs",
        run_name="simclr_example"
    )

    train_simclr(base_dataset, cfg)