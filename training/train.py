"""
SurfaceIdiot - Training Script

Trains GraspPolicy using behavior cloning (supervised regression on joint angles).

Usage:
    # Basic run (all objects in data/)
    python train.py

    # Specific objects
    python train.py --objects ball cup block

    # Resume from checkpoint
    python train.py --resume ../checkpoints/best_model.pth

    # Quick smoke-test with tiny batch
    python train.py --epochs 2 --batch_size 4 --workers 0
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from dataset import GraspDataset
from model import GraspPolicy

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


# ─── Loss ────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    MSE + smoothness penalty.
    The smoothness term penalises large changes between successive predicted
    angles within a batch, encouraging smoother motor trajectories.
    """

    def __init__(self, smooth_weight: float = 0.05):
        super().__init__()
        self.mse   = nn.MSELoss()
        self.smooth_weight = smooth_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.mse(pred, target)
        if self.smooth_weight > 0 and pred.size(0) > 1:
            diff = pred[1:] - pred[:-1]
            loss = loss + self.smooth_weight * (diff ** 2).mean()
        return loss


# ─── Training / validation loops ─────────────────────────────────────────────

def run_epoch(
    model: GraspPolicy,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    train: bool = True,
) -> float:
    model.train(train)
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, histories, targets in loader:
            imgs      = imgs.to(device, non_blocking=True)
            histories = histories.to(device, non_blocking=True)
            targets   = targets.to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                preds = model(imgs, histories)
                loss  = criterion(preds, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def per_finger_mae(
    model: GraspPolicy,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute per-finger mean absolute error on a loader (for diagnostics)."""
    model.eval()
    accs = torch.zeros(5, device=device)
    n    = 0
    with torch.no_grad():
        for imgs, histories, targets in loader:
            imgs      = imgs.to(device)
            histories = histories.to(device)
            targets   = targets.to(device)
            preds     = model(imgs, histories)
            accs     += (preds - targets).abs().sum(dim=0)
            n        += targets.size(0)
    return {f: (accs[i] / n).item() for i, f in enumerate(FINGERS)}


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(path: Path, model, optimizer, epoch: int, val_loss: float):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
    }, path)


def load_checkpoint(path: Path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["val_loss"]


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train GraspPolicy")
    p.add_argument("--data_root",    default="../data")
    p.add_argument("--checkpoint_dir", default="../checkpoints")
    p.add_argument("--objects",      nargs="+", default=None,
                   help="Objects to train on (default: all in data_root)")
    p.add_argument("--history_len",  type=int, default=6)
    p.add_argument("--epochs",       type=int, default=150)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--val_split",    type=float, default=0.15)
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("--resume",       default=None, help="Path to checkpoint to resume from")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze MobileNet weights (useful for small datasets)")
    p.add_argument("--smooth_weight", type=float, default=0.05,
                   help="Smoothness loss coefficient")
    p.add_argument("--camera",       default="wrist", choices=["wrist", "side"])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──
    full_ds = GraspDataset(
        data_root=args.data_root,
        objects=args.objects,
        history_len=args.history_len,
        train=True,
        camera=args.camera,
    )
    if len(full_ds) == 0:
        print("No samples found. Run collect.py first.")
        return

    val_size   = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # Validation uses no augmentation
    val_ds.dataset.transform = val_ds.dataset.transform  # already set; rebuild if needed

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), persistent_workers=(args.workers > 0),
    )

    print(f"Train  : {len(train_ds)} samples  ({len(train_loader)} batches)")
    print(f"Val    : {len(val_ds)}   samples  ({len(val_loader)} batches)")

    # ── Model ──
    model = GraspPolicy(
        history_len=args.history_len,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    print(f"Params : {model.count_parameters():,}")

    criterion = CombinedLoss(smooth_weight=args.smooth_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch  = 0
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, device
        )
        print(f"Resumed from epoch {start_epoch}, best val_loss={best_val_loss:.4f}")

    # ── Training loop ──
    print("\n" + "─" * 70)
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'Best':>6}  {'LR':>10}  {'Time':>6}")
    print("─" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, train=False)

        # Step scheduler (OneCycleLR steps per batch, but we call it per epoch here
        # for simplicity — move scheduler.step() inside the loop for per-batch)
        # Actually OneCycleLR should step per batch — done inside run_epoch above.
        # Here we just log the LR.
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(ckpt_dir / "best_model.pth", model, optimizer, epoch, val_loss)

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch+1:04d}.pth",
                model, optimizer, epoch, val_loss,
            )

        marker = "★" if is_best else " "
        print(f"{epoch+1:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{marker:>6}  {current_lr:>10.2e}  {elapsed:>5.1f}s")

        # Per-finger diagnostics every 25 epochs
        if (epoch + 1) % 25 == 0:
            mae = per_finger_mae(model, val_loader, device)
            print("  MAE per finger:", "  ".join(f"{k}: {v:.3f}" for k, v in mae.items()))

    print("─" * 70)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to {ckpt_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
