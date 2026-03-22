import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from read_labeled_data import *
from torch.utils.data import DataLoader, random_split


# ---------------------------------------------------
# 1. Encoder builder
# ---------------------------------------------------
def build_resnet18_encoder(pretrained=False):
    """
    Returns a ResNet-18 encoder that outputs a 512-d feature vector.
    Final fc layer is replaced by Identity.
    """
    model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()   # output: [B, 512]
    return model


# ---------------------------------------------------
# 2. Load SimCLR pretrained weights into encoder
# ---------------------------------------------------
def load_pretrained_encoder(encoder, ckpt_path, device="cpu", strict=False):
    """
    Load pretrained checkpoint into encoder.

    This function tries to be robust to common checkpoint formats:
    - full state_dict
    - {'state_dict': ...}
    - keys prefixed with 'encoder.'
    - keys prefixed with 'backbone.'
    - keys prefixed with 'module.'
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # extract model weights
    state_dict = ckpt["model"]

    new_state_dict = {}

    for k, v in state_dict.items():
        new_key = k

        # remove common prefixes from SimCLR training
        for prefix in ["module.", "encoder.", "backbone."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        # ❌ skip projection head (SimCLR-specific)
        if new_key.startswith("projector") or new_key.startswith("projection_head"):
            continue

        new_state_dict[new_key] = v

    missing, unexpected = encoder.load_state_dict(new_state_dict, strict=strict)

    print("✅ Loaded pretrained encoder from:", ckpt_path)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    return encoder


# ---------------------------------------------------
# 3. Class-specific attention MIL model
# ---------------------------------------------------
class MultiDefectAttentionModel(nn.Module):
    """
    Input:
        x: [B, 4, 3, 224, 224]

    Output:
        logits: [B, 3]
        attn_weights: [B, 3, 4]
    """
    def __init__(self, encoder, feature_dim=512, num_classes=3, attn_hidden=256, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Optional feature refinement after encoder
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Class-specific attention:
        # for each class, produce an attention score for each of the 4 blocks
        self.attn_mlp = nn.Sequential(
            nn.Linear(feature_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, num_classes)
        )

        # One classifier per class after attention pooling
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 1)
            )
            for _ in range(num_classes)
        ])

    def forward(self, x):
        """
        x: [B, 4, 3, 224, 224]
        """
        B, N, C, H, W = x.shape
        assert N == 4, f"Expected 4 blocks, got {N}"

        # Flatten block dimension into batch
        x = x.view(B * N, C, H, W)                 # [B*4, 3, 224, 224]

        feats = self.encoder(x)                    # [B*4, 512]
        feats = self.feature_proj(feats)           # [B*4, 512]
        feats = feats.view(B, N, self.feature_dim) # [B, 4, 512]

        # Attention scores per block, per class
        # [B, 4, 3]
        attn_scores = self.attn_mlp(feats)

        # Convert to [B, 3, 4] so each class has weights over 4 blocks
        attn_scores = attn_scores.permute(0, 2, 1)       # [B, 3, 4]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        logits = []
        for k in range(self.num_classes):
            # weighted sum of features across 4 blocks for class k
            # attn_weights[:, k, :] -> [B, 4]
            alpha = attn_weights[:, k, :].unsqueeze(-1)  # [B, 4, 1]
            pooled = torch.sum(alpha * feats, dim=1)     # [B, 512]

            logit_k = self.classifiers[k](pooled)        # [B, 1]
            logits.append(logit_k)

        logits = torch.cat(logits, dim=1)                # [B, 3]
        return logits, attn_weights


# ---------------------------------------------------
# 4. Freeze / unfreeze helpers
# ---------------------------------------------------
def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = True


def unfreeze_last_resnet_block(model):
    """
    Freeze all encoder params, then unfreeze layer4 only.
    Useful for gradual fine-tuning.
    """
    for p in model.encoder.parameters():
        p.requires_grad = False

    if hasattr(model.encoder, "layer4"):
        for p in model.encoder.layer4.parameters():
            p.requires_grad = True


# ---------------------------------------------------
# 5. Metrics
# ---------------------------------------------------
@torch.no_grad()
def compute_metrics_from_logits(logits, targets, threshold=0.5):
    """
    logits:  [B, 3]
    targets: [B, 3]
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    # element-wise accuracy across all labels
    label_acc = (preds == targets).float().mean().item()

    # exact-match accuracy: all 3 labels correct for a sample
    exact_match_acc = (preds == targets).all(dim=1).float().mean().item()

    # per-class accuracy
    per_class_acc = (preds == targets).float().mean(dim=0).cpu().tolist()

    return {
        "label_acc": label_acc,
        "exact_match_acc": exact_match_acc,
        "per_class_acc": per_class_acc
    }


# ---------------------------------------------------
# 6. One epoch train
# ---------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    total_samples = 0

    all_logits = []
    all_targets = []

    start_time = time.time()

    for batch, labels in loader:
        batch = batch.to(device, non_blocking=True)                 # [B, 4, 3, 224, 224]
        labels = labels.float().to(device, non_blocking=True)      # [B, 3]

        optimizer.zero_grad()

        logits, _ = model(batch)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        bs = batch.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    epoch_time = time.time() - start_time

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics_from_logits(all_logits, all_targets)

    return {
        "loss": running_loss / total_samples,
        "label_acc": metrics["label_acc"],
        "exact_match_acc": metrics["exact_match_acc"],
        "per_class_acc": metrics["per_class_acc"],
        "time": epoch_time
    }


# ---------------------------------------------------
# 7. Validation
# ---------------------------------------------------
@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_logits = []
    all_targets = []

    for batch, labels in loader:
        batch = batch.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        logits, _ = model(batch)
        loss = criterion(logits, labels)

        bs = batch.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics_from_logits(all_logits, all_targets)

    return {
        "loss": running_loss / total_samples,
        "label_acc": metrics["label_acc"],
        "exact_match_acc": metrics["exact_match_acc"],
        "per_class_acc": metrics["per_class_acc"],
    }


# ---------------------------------------------------
# 8. Fine-tuning driver
# ---------------------------------------------------
def finetune_model(
    train_loader,
    val_loader,
    simclr_ckpt_path,
    num_epochs=20,
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    freeze_backbone_epochs=0,
    unfreeze_strategy="all",   # "all" or "last_block"
    save_path="best_multidefect_model.pt",
    pos_weight=None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build encoder and load SimCLR pretrained weights
    encoder = build_resnet18_encoder(pretrained=False)
    encoder = load_pretrained_encoder(encoder, simclr_ckpt_path, device=device, strict=False)

    # Build downstream model
    model = MultiDefectAttentionModel(
        encoder=encoder,
        feature_dim=512,
        num_classes=3,
        attn_hidden=256,
        dropout=0.2
    ).to(device)

    # Optional initial freezing
    if freeze_backbone_epochs > 0:
        if unfreeze_strategy == "last_block":
            unfreeze_last_resnet_block(model)
        else:
            freeze_encoder(model)
    else:
        unfreeze_encoder(model)

    # Loss
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Only optimize trainable params
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(num_epochs):
        # Unfreeze after some warmup epochs
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            if unfreeze_strategy == "last_block":
                # first stage was only layer4 trainable; now full unfreeze
                unfreeze_encoder(model)
            else:
                unfreeze_encoder(model)

            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )
            print(f"Epoch {epoch+1}: encoder unfrozen.")

        train_stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_stats = validate_one_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "train_label_acc": train_stats["label_acc"],
            "train_exact_match_acc": train_stats["exact_match_acc"],
            "val_loss": val_stats["loss"],
            "val_label_acc": val_stats["label_acc"],
            "val_exact_match_acc": val_stats["exact_match_acc"],
            "train_per_class_acc": train_stats["per_class_acc"],
            "val_per_class_acc": val_stats["per_class_acc"],
            "epoch_time": train_stats["time"]
        })

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_stats['loss']:.4f} | "
            f"Train Label Acc: {train_stats['label_acc']:.4f} | "
            f"Train Exact Acc: {train_stats['exact_match_acc']:.4f} || "
            f"Val Loss: {val_stats['loss']:.4f} | "
            f"Val Label Acc: {val_stats['label_acc']:.4f} | "
            f"Val Exact Acc: {val_stats['exact_match_acc']:.4f} | "
            f"Time: {train_stats['time']:.1f}s"
        )

        print("  Train per-class acc:", [round(x, 4) for x in train_stats["per_class_acc"]])
        print("  Val   per-class acc:", [round(x, 4) for x in val_stats["per_class_acc"]])

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": best_state,
                "best_val_loss": best_val_loss,
                "history": history
            }, save_path)
            print(f"  Saved best model to {save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def predict_loader(model, loader, device="cuda", threshold=0.5):
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    all_probs = []
    all_preds = []
    all_labels = []

    for batch, labels in loader:
        batch = batch.to(device)
        logits, attn_weights = model(batch)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    return (
        torch.cat(all_probs, dim=0),
        torch.cat(all_preds, dim=0),
        torch.cat(all_labels, dim=0),
    )

if __name__ == "__main__":
    print("SFT")

    device = "cuda"

    dataset = LabeledDefectDataset("Data/train_labeled", cropped=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)   # for reproducibility
    )

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model, history = finetune_model(
        train_loader=train_loader,
        val_loader=val_loader,
        simclr_ckpt_path="simclr_runs/simclr_example/checkpoints/best.pt",  # change to your checkpoint
        num_epochs=30,
        lr=1e-4,
        weight_decay=1e-4,
        device=device,
        freeze_backbone_epochs=3,   # freeze first 3 epochs, then unfreeze
        unfreeze_strategy="all",    # or "last_block"
        save_path="best_defect_attention_model.pt",
        pos_weight=None             # or e.g. [2.0, 3.5, 1.8] if imbalanced
    )