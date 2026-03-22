import os
import csv
import math
import time
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from tqdm import tqdm

from cropped_data_loader import build_dataloader


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    # data
    batch_size: int = 128              # per GPU
    num_workers: int = 8
    image_size: int = 224
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True

    # optimization
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.2

    # model
    backbone_name: str = "resnet18"
    pretrained: bool = True
    projection_dim: int = 128
    hidden_dim: int = 512

    # system
    use_amp: bool = True
    seed: int = 42
    compile_model: bool = False   # set True if PyTorch 2.x and stable on your env

    # paths
    save_dir: str = "./simclr_runs"
    run_name: str = "simclr_ddp"
    resume_path: str = ""

    # data source
    data_dir: str = "Data/cropped/chunked"
    pattern: str = "X_unlabeled_root_*.pt"

    # logging
    log_interval: int = 20
    save_every: int = 1


# ============================================================
# Distributed helpers
# ============================================================

def ddp_setup():
    """
    torchrun sets:
      RANK, WORLD_SIZE, LOCAL_RANK
    """
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_seed_per_rank(seed: int, rank: int):
    set_seed(seed + rank)


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
            writer.writerow(["epoch", "train_loss", "lr", "epoch_time_sec"])


def append_csv_log(csv_path: str, epoch: int, train_loss: float, lr: float, epoch_time_sec: float):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, lr, epoch_time_sec])


# ============================================================
# SimCLR augmentations
# ============================================================

class SimCLRTransform:
    """
    Tensor-friendly transform path.
    Assumes input is PIL or torch Tensor [C,H,W].
    Removes repeated ToPILImage() overhead.
    """
    def __init__(self, image_size=224):
        color_jitter = transforms.ColorJitter(
            brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
        )

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.dtype == torch.uint8:
                pass
            elif x.max() <= 1.0:
                x = (x * 255.0).clamp(0, 255).to(torch.uint8)
            else:
                x = x.clamp(0, 255).to(torch.uint8)

        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


# ============================================================
# Dataset wrapper
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
    def __init__(self, backbone_name="resnet18", pretrained=True, projection_dim=128, hidden_dim=512):
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

        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = ProjectionHead(
            in_dim=feat_dim,
            hidden_dim=hidden_dim,
            out_dim=projection_dim
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


# ============================================================
# Global gather for SimCLR
# ============================================================

@torch.no_grad()
def concat_all_gather(x):
    """
    Gather tensor from all ranks.
    Returns concatenated tensor of shape [world_size * B, ...]
    """
    if not dist.is_initialized():
        return x

    world_size = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


class NTXentLossGlobal(nn.Module):
    """
    SimCLR NT-Xent using GLOBAL negatives across all GPUs.
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        local z1, z2: [B, D]
        gathered -> [global_B, D]
        """
        device = z1.device
        local_batch = z1.size(0)

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        z1_all = concat_all_gather(z1)
        z2_all = concat_all_gather(z2)

        z = torch.cat([z1_all, z2_all], dim=0)   # [2G, D]
        global_batch = z1_all.size(0)

        sim = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * global_batch, device=device, dtype=torch.bool)
        sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

        targets = torch.arange(global_batch, 2 * global_batch, device=device)
        targets = torch.cat([targets, torch.arange(0, global_batch, device=device)], dim=0)

        loss_all = F.cross_entropy(sim, targets, reduction="none")

        # only keep this rank's samples
        start = rank * local_batch
        end = start + local_batch

        idx_local = torch.arange(start, end, device=device)
        idx_local_2 = idx_local + global_batch
        idx_keep = torch.cat([idx_local, idx_local_2], dim=0)

        loss = loss_all[idx_keep].mean()
        return loss


# ============================================================
# Scheduler
# ============================================================

def build_cosine_scheduler(optimizer, total_epochs, warmup_epochs=5, min_lr=1e-6):
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_loss):
    model_to_save = model.module if hasattr(model, "module") else model
    ckpt = {
        "epoch": epoch,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_loss": best_loss,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    epoch = ckpt.get("epoch", -1)
    best_loss = ckpt.get("best_loss", float("inf"))
    return epoch, best_loss


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, cfg):
    model.train()
    running_loss = torch.zeros(1, device=device)
    start_time = time.time()

    use_tqdm = is_main_process()
    iterator = tqdm(loader, disable=not use_tqdm)

    for step, (x1, x2) in enumerate(iterator):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)

        if scaler is not None and cfg.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.detach()

        if is_main_process() and (step + 1) % cfg.log_interval == 0:
            avg_so_far = (running_loss / (step + 1)).item()
            iterator.set_description(f"Epoch {epoch} Loss {avg_so_far:.4f}")

    # average loss across ranks
    if dist.is_initialized():
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        running_loss /= dist.get_world_size()

    epoch_loss = (running_loss / max(1, len(loader))).item()
    epoch_time = time.time() - start_time
    return epoch_loss, epoch_time


# ============================================================
# Main
# ============================================================

def main():
    cfg = TrainConfig(
        batch_size=128,
        epochs=20,
        lr=3e-4,
        backbone_name="resnet18",
        pretrained=True,
        projection_dim=128,
        hidden_dim=128,
        save_dir="./simclr_runs",
        run_name="simclr_ddp_gl",
        num_workers=8,
        data_dir="Data/cropped/chunked",
        pattern="X_unlabeled_root_*.pt",
        compile_model=False,
    )

    rank, world_size, local_rank, device = ddp_setup()

    set_seed_per_rank(cfg.seed, rank)
    torch.backends.cudnn.benchmark = True

    run_dir = os.path.join(cfg.save_dir, cfg.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    if is_main_process():
        ensure_dir(run_dir)
        ensure_dir(ckpt_dir)
        save_config(cfg, run_dir)
        write_csv_log_header(os.path.join(run_dir, "train_log.csv"))

    dist.barrier()

    # --------------------------------------------------------
    # Build dataset
    # We call your existing build_dataloader only to obtain the dataset.
    # Then we rebuild the DataLoader with DistributedSampler.
    # --------------------------------------------------------
    simclr_transform = SimCLRTransform(image_size=cfg.image_size)

    dataset, _ = build_dataloader(
        data_dir=cfg.data_dir,
        pattern=cfg.pattern,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        transform=simclr_transform,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,    # important for SimCLR global gather shape consistency
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = SimCLRModel(
        backbone_name=cfg.backbone_name,
        pretrained=cfg.pretrained,
        projection_dim=cfg.projection_dim,
        hidden_dim=cfg.hidden_dim
    ).to(device)

    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )

    criterion = NTXentLossGlobal(temperature=cfg.temperature)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    scheduler = build_cosine_scheduler(
        optimizer,
        total_epochs=cfg.epochs,
        warmup_epochs=5,
        min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    start_epoch = 0
    best_loss = float("inf")

    if cfg.resume_path and os.path.isfile(cfg.resume_path):
        if is_main_process():
            print(f"Resuming from checkpoint: {cfg.resume_path}")
        last_epoch, best_loss = load_checkpoint(
            cfg.resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = last_epoch + 1

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        sampler.set_epoch(epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        if is_main_process():
            print("=" * 80)
            print(f"Epoch {epoch}/{cfg.epochs - 1} | LR: {current_lr:.8f}")

        train_loss, epoch_time = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            cfg=cfg,
        )

        scheduler.step()

        if is_main_process():
            print(
                f"Epoch {epoch} finished | "
                f"Train Loss: {train_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            csv_log_path = os.path.join(run_dir, "train_log.csv")
            append_csv_log(
                csv_log_path,
                epoch=epoch,
                train_loss=train_loss,
                lr=current_lr,
                epoch_time_sec=epoch_time
            )

            latest_path = os.path.join(ckpt_dir, "latest.pt")
            save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, best_loss)

            if train_loss < best_loss:
                best_loss = train_loss
                best_path = os.path.join(ckpt_dir, "best.pt")
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_loss)
                print(f"Saved new best checkpoint to: {best_path}")

            if (epoch + 1) % cfg.save_every == 0:
                epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
                save_checkpoint(epoch_path, model, optimizer, scheduler, scaler, epoch, best_loss)
                print(f"Saved checkpoint to: {epoch_path}")

    if is_main_process():
        print("=" * 80)
        print("Training finished.")
        print(f"Best training loss: {best_loss:.4f}")
        print(f"Run directory: {run_dir}")

    cleanup_ddp()


if __name__ == "__main__":
    main()