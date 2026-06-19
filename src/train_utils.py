import time
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from config import NUM_CLASSES, IGNORE_LABEL, CLASS_NAMES


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=IGNORE_LABEL):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred_soft = torch.softmax(pred, dim=1)
        valid = (target != self.ignore_index).unsqueeze(1)
        target_onehot = torch.zeros_like(pred_soft)
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        target_onehot.scatter_(1, target_clamped.unsqueeze(1), 1)
        target_onehot = target_onehot * valid
        pred_soft = pred_soft * valid
        intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CEDiceLoss(torch.nn.Module):
    def __init__(self, weight=None, ignore_index=IGNORE_LABEL, dice_weight=0.5, label_smoothing=0.0):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, label_smoothing=label_smoothing
        )
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return (1 - self.dice_weight) * self.ce(pred, target) + self.dice_weight * self.dice(pred, target)


def build_criterion(loss_fn, class_weights=None, device=None, label_smoothing=0.0):
    w = None
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32)
        if device is not None:
            w = w.to(device)
    if loss_fn == "ce":
        return torch.nn.CrossEntropyLoss(
            weight=w, ignore_index=IGNORE_LABEL, label_smoothing=label_smoothing
        )
    elif loss_fn == "ce_dice":
        return CEDiceLoss(weight=w, ignore_index=IGNORE_LABEL, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss_fn: {loss_fn}")


def detect_amp_dtype():
    if not torch.cuda.is_available():
        return None
    for dtype in [torch.float16, torch.bfloat16]:
        try:
            conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
            x = torch.randn(1, 3, 32, 32, device="cuda")
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = conv(x)
            del conv, x
            torch.cuda.empty_cache()
            return dtype
        except Exception:
            torch.cuda.empty_cache()
    return None


# --- Global (dataset-level) segmentation metrics ------------------------------
# Dice/IoU are accumulated as raw per-class counts across the WHOLE loader and
# the ratio is taken once at the end (a "ratio of sums", i.e. micro/global
# aggregation). This is the same scheme the notebook test cell uses, so train,
# valid and test all report the SAME metric and are directly comparable.
#
# The previous per-batch averaging ("mean of per-batch ratios") systematically
# deflated Dice on rare classes and depended on batch size -- e.g. it reported
# valid mDice 0.5603 @bs16 where the global value is 0.7504 on identical
# predictions. See scripts/verify_dice_discrepancy.py for the reproduction.

def _new_counts(device):
    return {
        "inter": torch.zeros(NUM_CLASSES, device=device),
        "parea": torch.zeros(NUM_CLASSES, device=device),
        "tarea": torch.zeros(NUM_CLASSES, device=device),
        "union": torch.zeros(NUM_CLASSES, device=device),
        "correct": torch.zeros((), device=device),
        "total": torch.zeros((), device=device),
    }


def _add_counts(acc, pred_logits, target, num_classes=NUM_CLASSES):
    """Accumulate one batch's per-class pixel counts (additive across batches)."""
    pred = pred_logits.argmax(dim=1)
    valid = target != IGNORE_LABEL
    acc["correct"] += ((pred == target) & valid).sum()
    acc["total"] += valid.sum()
    for c in range(num_classes):
        pc = (pred == c) & valid
        tc = (target == c) & valid
        acc["inter"][c] += (pc & tc).sum()
        acc["parea"][c] += pc.sum()
        acc["tarea"][c] += tc.sum()
        acc["union"][c] += (pc | tc).sum()


def _counts_to_metrics(acc, loss_sum, n, smooth=1e-6):
    inter = acc["inter"].cpu().numpy()
    parea = acc["parea"].cpu().numpy()
    tarea = acc["tarea"].cpu().numpy()
    union = acc["union"].cpu().numpy()
    dice = (2 * inter + smooth) / (parea + tarea + smooth)
    iou = (inter + smooth) / (union + smooth)
    return {
        "loss": loss_sum / n,
        "dice_per_class": dice.tolist(),
        "iou_per_class": iou.tolist(),
        "mean_dice": float(dice.mean()),
        "mean_iou": float(iou.mean()),
        "pixel_accuracy": float((acc["correct"] / (acc["total"] + 1e-6)).item()),
    }


def train_one_epoch(model, loader, criterion, optimizer, device, amp_dtype=None,
                    scaler=None, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    acc = _new_counts(device)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if scaler is not None and amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(images)
                loss = criterion(out, masks)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, masks)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        with torch.no_grad():
            _add_counts(acc, out.float(), masks)

    return _counts_to_metrics(acc, total_loss, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    acc = _new_counts(device)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        out = model(images)
        total_loss += criterion(out, masks).item() * images.size(0)
        _add_counts(acc, out, masks)

    return _counts_to_metrics(acc, total_loss, len(loader.dataset))


@torch.no_grad()
def predict_samples(model, dataset, indices, device):
    model.eval()
    results = []
    for idx in indices:
        image, mask = dataset[idx]
        pred = model(image.unsqueeze(0).to(device))
        pred_class = pred.argmax(dim=1).squeeze(0).cpu().numpy()
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        results.append({"image": img_np, "mask": mask.numpy(), "pred": pred_class})
    return results


def _row(epoch, phase, m, lr=None, elapsed=None):
    row = {"epoch": epoch, "phase": phase, "loss": m["loss"],
           "mean_dice": m["mean_dice"], "mean_iou": m["mean_iou"],
           "pixel_accuracy": m["pixel_accuracy"], "lr": lr, "time_sec": elapsed}
    for name, d in zip(CLASS_NAMES, m["dice_per_class"]):
        row[f"dice_{name}"] = d
    for name, v in zip(CLASS_NAMES, m["iou_per_class"]):
        row[f"iou_{name}"] = v
    return row


def build_scheduler(optimizer, epochs, warmup_epochs=5, steps_per_epoch=1):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs * steps_per_epoch
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs) * steps_per_epoch
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[warmup_epochs * steps_per_epoch]
    )


def train_model(
    model, train_loader, valid_loader, criterion, optimizer, scheduler,
    device, epochs, save_dir, amp_dtype=None, early_stopping_patience=None,
    grad_clip=1.0,
):
    save_dir = Path(save_dir)
    ckpt_dir = save_dir / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    scaler = torch.GradScaler() if amp_dtype is not None else None

    history = {
        "train_loss": [], "train_dice": [], "train_iou": [], "train_acc": [],
        "valid_loss": [], "valid_dice": [], "valid_iou": [], "valid_acc": [],
    }
    log_rows = []
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        tm = train_one_epoch(model, train_loader, criterion, optimizer, device,
                             amp_dtype, scaler, grad_clip)
        vm = evaluate(model, valid_loader, criterion, device)

        if scheduler:
            scheduler.step()

        elapsed = time.time() - t0

        history["train_loss"].append(tm["loss"])
        history["train_dice"].append(tm["mean_dice"])
        history["train_iou"].append(tm["mean_iou"])
        history["train_acc"].append(tm["pixel_accuracy"])
        history["valid_loss"].append(vm["loss"])
        history["valid_dice"].append(vm["mean_dice"])
        history["valid_iou"].append(vm["mean_iou"])
        history["valid_acc"].append(vm["pixel_accuracy"])

        log_rows.append(_row(epoch, "train", tm, lr, elapsed))
        log_rows.append(_row(epoch, "valid", vm, lr, elapsed))
        pd.DataFrame(log_rows).to_csv(save_dir / "training_log.csv", index=False)

        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")

        is_best = vm["mean_dice"] > best_dice
        if is_best:
            best_dice = vm["mean_dice"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        else:
            patience_counter += 1

        print(f"Epoch [{epoch:3d}/{epochs}] | "
              f"Train Loss: {tm['loss']:.4f}  Dice: {tm['mean_dice']:.4f}  IoU: {tm['mean_iou']:.4f} | "
              f"Valid Loss: {vm['loss']:.4f}  Dice: {vm['mean_dice']:.4f}  IoU: {vm['mean_iou']:.4f} | "
              f"LR: {lr:.2e} | {elapsed:.0f}s"
              f"{' [best]' if is_best else ''}")

        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    history["best_epoch"] = best_epoch
    history["best_dice"] = best_dice
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best valid mDice: {best_dice:.4f} at epoch {best_epoch}")
    return history


def per_image_metrics(pred, mask, num_classes=NUM_CLASSES, smooth=1e-6):
    valid = mask != IGNORE_LABEL
    dice_scores = []
    iou_scores = []
    for c in range(num_classes):
        pc = (pred == c) & valid
        tc = (mask == c) & valid
        inter = float((pc & tc).sum())
        dice_union = float(pc.sum() + tc.sum())
        iou_union = float((pc | tc).sum())
        dice_scores.append((2 * inter + smooth) / (dice_union + smooth))
        iou_scores.append((inter + smooth) / (iou_union + smooth))
    return {
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
        "dice_per_class": {CLASS_NAMES[c]: dice_scores[c] for c in range(num_classes)},
        "iou_per_class": {CLASS_NAMES[c]: iou_scores[c] for c in range(num_classes)},
    }


def select_diverse_samples(dataset, n=4):
    class_coverage = []
    for idx in range(len(dataset)):
        _, mask = dataset[idx]
        mask_np = mask.numpy() if hasattr(mask, 'numpy') else mask
        classes_present = set(np.unique(mask_np)) - {IGNORE_LABEL}
        class_coverage.append((idx, classes_present))

    selected, covered = [], set()
    rare_first = sorted(class_coverage, key=lambda x: (-len({3, 2} & x[1]), -len(x[1])))

    for idx, classes in rare_first:
        if len(selected) >= n:
            break
        if classes - covered:
            selected.append(idx)
            covered |= classes

    if len(selected) < n:
        remaining = sorted(
            [(i, c) for i, c in class_coverage if i not in selected],
            key=lambda x: -len(x[1])
        )
        for idx, _ in remaining:
            if len(selected) >= n:
                break
            selected.append(idx)

    return selected
