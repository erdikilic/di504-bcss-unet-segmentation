import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

from config import CLASS_NAMES, CLASS_COLORS, NUM_CLASSES, IGNORE_LABEL

PALETTE = ["#E63946", "#2A9D8F", "#264653", "#E9C46A", "#8D99AE"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def _finish(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    plt.show()
    plt.close(fig)


def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = CLASS_COLORS[c]
    rgb[mask == IGNORE_LABEL] = (0, 0, 0)
    return rgb


def tp_fn_fp_overlay(mask, pred):
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    valid = mask != IGNORE_LABEL

    for c in range(NUM_CLASSES):
        tp_c = (pred == c) & (mask == c) & valid
        overlay[tp_c] = CLASS_COLORS[c]

    wrong = (pred != mask) & valid
    for c in range(NUM_CLASSES):
        fn_c = (mask == c) & wrong
        overlay[fn_c] = (255, 18, 0)
        fp_c = (pred == c) & wrong
        overlay[fp_c] = (0, 254, 0)

    return overlay


def plot_single_prediction(r, save_path=None, idx=None):
    dice = r.get("dice", None)
    iou = r.get("iou", None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(r["image"])
    axes[0].set_title("Input")
    axes[1].imshow(mask_to_rgb(r["mask"]))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(mask_to_rgb(r["pred"]))
    axes[2].set_title("Prediction")

    for ax in axes:
        ax.axis("off")

    title_parts = []
    if idx is not None:
        title_parts.append(f"Sample: {idx}")
    if dice is not None:
        title_parts.append(f"Dice: {dice:.3f}")
    if iou is not None:
        title_parts.append(f"IoU: {iou:.3f}")

    per_class_d = r.get("dice_per_class", {})
    per_class_i = r.get("iou_per_class", {})
    mask_vals = r["mask"]
    present = set(np.unique(mask_vals)) - {IGNORE_LABEL}

    class_parts = []
    for ci, name in enumerate(CLASS_NAMES):
        if ci in present and name in per_class_d:
            class_parts.append(f"{name}: D={per_class_d[name]:.2f} I={per_class_i.get(name, 0):.2f}")

    title_line1 = " | ".join(title_parts)
    title_line2 = "  ".join(class_parts) if class_parts else ""
    full_title = f"{title_line1}\n{title_line2}" if title_line2 else title_line1
    fig.suptitle(full_title, fontsize=13)

    handles = [mpatches.Patch(color=np.array(CLASS_COLORS[ci]) / 255, label=name)
               for ci, name in enumerate(CLASS_NAMES) if ci in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True, fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    _finish(fig, save_path)


def plot_predictions(results, save_dir=None, prefix="pred"):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for i, r in enumerate(results):
        sp = os.path.join(save_dir, f"{prefix}_{i:03d}.png") if save_dir else None
        plot_single_prediction(r, save_path=sp, idx=i)


def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix", already_filtered=False):
    from sklearn.metrics import confusion_matrix
    if not already_filtered:
        valid = y_true != IGNORE_LABEL
        y_true = y_true[valid]
        y_pred = y_pred[valid]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]:,})",
                    ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    _finish(fig, save_path)


def plot_per_class_metrics(dice_per_class, iou_per_class, save_path=None, title=None):
    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, dice_per_class, width, label="Dice", color=PALETTE[1], edgecolor="white")
    bars2 = ax.bar(x + width / 2, iou_per_class, width, label="IoU", color=PALETTE[0], edgecolor="white")

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    _finish(fig, save_path)


def classification_report_df(y_true, y_pred, already_filtered=False):
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    if not already_filtered:
        valid = y_true != IGNORE_LABEL
        y_true = y_true[valid]
        y_pred = y_pred[valid]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    rows = []
    for i in range(NUM_CLASSES):
        rows.append({"Class": CLASS_NAMES[i], "Precision": precision[i], "Recall": recall[i],
                      "F1": f1[i], "Support": int(support[i])})
    rows.append({"Class": "Macro avg", "Precision": precision.mean(), "Recall": recall.mean(),
                  "F1": f1.mean(), "Support": int(support.sum())})
    return pd.DataFrame(rows).set_index("Class")


def plot_per_image_dice_histogram(per_image_dice, save_path=None, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(per_image_dice, bins=30, color=PALETTE[1], edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(per_image_dice), color=PALETTE[0], linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(per_image_dice):.4f}")
    axes[0].axvline(np.median(per_image_dice), color=PALETTE[2], linestyle=":", linewidth=2,
                    label=f"Median: {np.median(per_image_dice):.4f}")
    axes[0].set_xlabel("Mean Dice Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution")
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot(per_image_dice, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[1], alpha=0.6),
                    medianprops=dict(color=PALETTE[0], linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    axes[1].set_ylabel("Mean Dice Score")
    axes[1].set_title("Box Plot")
    axes[1].set_xticklabels([""])
    axes[1].grid(axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, save_path)


def plot_best_worst_predictions(results_with_dice, n=4, save_dir=None):
    sorted_results = sorted(results_with_dice, key=lambda x: x["dice"])
    worst = sorted_results[:n]
    best = sorted_results[-n:][::-1]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for label, samples in [("Best", best), ("Worst", worst)]:
        for i, r in enumerate(samples):
            sp = os.path.join(save_dir, f"{label.lower()}_{i:02d}.png") if save_dir else None
            plot_single_prediction(r, save_path=sp, idx=f"{label} #{i+1}")


def plot_training_per_class(history_json_path, save_path=None, title=None):
    import pandas as pd
    df = pd.read_csv(history_json_path.replace("history.json", "training_log.csv"))
    valid_df = df[df["phase"] == "valid"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(CLASS_NAMES):
        dcol = f"dice_{name}"
        icol = f"iou_{name}"
        if dcol in valid_df.columns:
            ax.plot(valid_df["epoch"], valid_df[dcol], color=PALETTE[i],
                    linewidth=2, linestyle="-", label=f"{name.capitalize()} Dice")
        if icol in valid_df.columns:
            ax.plot(valid_df["epoch"], valid_df[icol], color=PALETTE[i],
                    linewidth=1.5, linestyle="--", alpha=0.7, label=f"{name.capitalize()} IoU")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.legend(loc="lower center", ncol=5, frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))
    if title:
        ax.set_title(title)
    plt.tight_layout()
    _finish(fig, save_path)


def _boundary_stats(y_true, y_pred, dilation=2):
    valid = y_true != IGNORE_LABEL
    y_true_v = y_true.copy()
    y_true_v[~valid] = 0

    edges = np.zeros_like(y_true, dtype=bool)
    for c in range(NUM_CLASSES):
        mask_c = y_true_v == c
        dilated = ndimage.binary_dilation(mask_c, iterations=dilation)
        eroded = ndimage.binary_erosion(mask_c, iterations=dilation)
        edges |= dilated & ~eroded

    edges &= valid
    total = int(edges.sum())
    if total == 0:
        return 0, 0
    correct = int((y_pred[edges] == y_true[edges]).sum())
    return correct, total


def compute_boundary_accuracy(y_true, y_pred, dilation=2):
    correct, total = _boundary_stats(y_true, y_pred, dilation)
    return 1.0 if total == 0 else correct / total


def plot_boundary_accuracy(y_true_images, y_pred_images, save_path=None):
    per_image_boundary = []
    total_correct = 0
    total_pixels = 0
    for gt, pred in zip(y_true_images, y_pred_images):
        c, t = _boundary_stats(gt, pred)
        per_image_boundary.append(c / t if t > 0 else 1.0)
        total_correct += c
        total_pixels += t
    overall = total_correct / total_pixels if total_pixels > 0 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(per_image_boundary, bins=30, color=PALETTE[1], edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(per_image_boundary), color=PALETTE[0], linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(per_image_boundary):.4f}")
    axes[0].set_xlabel("Boundary Accuracy")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Per-Image Distribution")
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", alpha=0.3)

    bars = axes[1].bar(["Overall", "Mean\nper-image"],
                       [overall, np.mean(per_image_boundary)],
                       color=[PALETTE[1], PALETTE[0]], edgecolor="white", width=0.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Summary")
    axes[1].grid(axis="y", alpha=0.3)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{bar.get_height():.4f}", ha="center", fontsize=10)

    plt.suptitle("Boundary Accuracy", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, save_path)
    return {"overall": overall, "mean_per_image": float(np.mean(per_image_boundary))}


def plot_comparison_table(all_metrics, save_path=None):
    variants = list(all_metrics.keys())
    dice_scores = [all_metrics[v]["mean_dice"] for v in variants]
    iou_scores = [all_metrics[v]["mean_iou"] for v in variants]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, dice_scores, width, label="Mean Dice", color=PALETTE[1], edgecolor="white")
    bars2 = ax.bar(x + width / 2, iou_scores, width, label="Mean IoU", color=PALETTE[0], edgecolor="white")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9)
    ax.legend(frameon=True)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    _finish(fig, save_path)


def plot_class_distribution(data_dir, splits=("train", "valid", "test"), save_path=None):
    from PIL import Image as PILImage
    import pandas as pd
    from pathlib import Path

    rows = []
    for split in splits:
        mask_dir = Path(data_dir) / split / "masks"
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for f in sorted(os.listdir(mask_dir)):
            mask = np.array(PILImage.open(mask_dir / f))
            for c in range(NUM_CLASSES):
                counts[c] += np.sum(mask == c)
        for c in range(NUM_CLASSES):
            rows.append({"Split": split, "Class": CLASS_NAMES[c], "Pixels": int(counts[c])})

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for si, split in enumerate(splits):
        sub = df[df["Split"] == split]
        axes[0].bar(np.arange(NUM_CLASSES) + si * 0.25, sub["Pixels"].values / 1e6,
                    width=0.25, label=split)
    axes[0].set_xticks(np.arange(NUM_CLASSES) + 0.25)
    axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_ylabel("Pixels (millions)")
    axes[0].set_title("Pixel Count per Class")
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    total_per_split = df.groupby("Split")["Pixels"].sum()
    for si, split in enumerate(splits):
        sub = df[df["Split"] == split]
        pcts = sub["Pixels"].values / total_per_split[split] * 100
        axes[1].bar(np.arange(NUM_CLASSES) + si * 0.25, pcts, width=0.25, label=split)
    axes[1].set_xticks(np.arange(NUM_CLASSES) + 0.25)
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Class Distribution (%)")
    axes[1].legend(frameon=True)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("Dataset Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, save_path)


def plot_augmented_samples(dataset_raw, dataset_aug, n=4, save_path=None):
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        idx = i * (len(dataset_raw) // n)
        img_raw, mask_raw = dataset_raw[idx]
        img_aug, mask_aug = dataset_aug[idx]

        if hasattr(img_raw, 'numpy'):
            img_r = img_raw.permute(1, 2, 0).numpy()
            img_r = img_r * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_r = np.clip(img_r * 255, 0, 255).astype(np.uint8)
        else:
            img_r = img_raw
        img_a = img_aug.permute(1, 2, 0).numpy()
        img_a = img_a * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_a = np.clip(img_a * 255, 0, 255).astype(np.uint8)

        mask_r = mask_raw.numpy() if hasattr(mask_raw, 'numpy') else mask_raw
        mask_a = mask_aug.numpy() if hasattr(mask_aug, 'numpy') else mask_aug

        axes[i, 0].imshow(img_r)
        axes[i, 1].imshow(mask_to_rgb(mask_r))
        axes[i, 2].imshow(img_a)
        axes[i, 3].imshow(mask_to_rgb(mask_a))
        if i == 0:
            axes[i, 0].set_title("Original")
            axes[i, 1].set_title("Original Mask")
            axes[i, 2].set_title("Augmented")
            axes[i, 3].set_title("Augmented Mask")

    for ax in axes.ravel():
        ax.axis("off")
    plt.suptitle("Augmentation Examples", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, save_path)


def plot_rgb_histogram(dataset, n=100, save_path=None):
    rs, gs, bs = [], [], []
    for i in range(min(n, len(dataset))):
        img, _ = dataset[i]
        if hasattr(img, 'numpy'):
            img = img.permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img * 255, 0, 255)
        rs.extend(img[:, :, 0].flatten()[::10].tolist())
        gs.extend(img[:, :, 1].flatten()[::10].tolist())
        bs.extend(img[:, :, 2].flatten()[::10].tolist())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rs, bins=50, alpha=0.5, label="Red", color="red", density=True)
    ax.hist(gs, bins=50, alpha=0.5, label="Green", color="green", density=True)
    ax.hist(bs, bins=50, alpha=0.5, label="Blue", color="blue", density=True)
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Density")
    ax.set_title("RGB Channel Distribution")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _finish(fig, save_path)


def plot_lr_schedule(history_path, save_path=None):
    import pandas as pd
    df = pd.read_csv(history_path)
    train_df = df[df["phase"] == "train"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_df["epoch"], train_df["lr"], color=PALETTE[2], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _finish(fig, save_path)


def plot_overfitting_gap(history, save_path=None, title=None):
    gap = np.array(history["train_dice"]) - np.array(history["valid_dice"])
    epochs = range(1, len(gap) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(epochs, 0, gap, alpha=0.3, color=PALETTE[0])
    ax.plot(epochs, gap, color=PALETTE[0], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Dice − Valid Dice")
    ax.set_title(title or "Overfitting Analysis")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _finish(fig, save_path)




def plot_class_confusion_analysis(y_true, y_pred, save_path=None, already_filtered=False):
    from sklearn.metrics import confusion_matrix
    if not already_filtered:
        valid = y_true != IGNORE_LABEL
        y_true = y_true[valid]
        y_pred = y_pred[valid]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((f"{CLASS_NAMES[i]} → {CLASS_NAMES[j]}", cm[i, j]))
    pairs.sort(key=lambda x: -x[1])
    top = pairs[:10]

    fig, ax = plt.subplots(figsize=(10, 5))
    if top:
        labels, values = zip(*top)
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, [v / 1e6 for v in values], color=PALETTE[0], edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    ax.set_xlabel("Misclassified Pixels (millions)")
    ax.set_title("Top Class Confusions")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    _finish(fig, save_path)


def plot_confidence_heatmap(model, dataset, indices, device, save_path=None):
    import torch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=0)
        max_prob = probs.max(axis=0)
        img_np = image.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        axes[i, 0].imshow(img_np)
        axes[i, 0].axis("off")

        im1 = axes[i, 1].imshow(max_prob, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 1].axis("off")
        div1 = make_axes_locatable(axes[i, 1])
        fig.colorbar(im1, cax=div1.append_axes("right", size="5%", pad=0.05))

        im2 = axes[i, 2].imshow(entropy, cmap="hot")
        axes[i, 2].axis("off")
        div2 = make_axes_locatable(axes[i, 2])
        fig.colorbar(im2, cax=div2.append_axes("right", size="5%", pad=0.05))

        if i == 0:
            axes[i, 0].set_title("Input")
            axes[i, 1].set_title("Confidence (max prob)")
            axes[i, 2].set_title("Entropy (uncertainty)")

    plt.suptitle("Prediction Confidence Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, save_path)


def model_summary_df(model_name, params, history, test_metrics):
    import pandas as pd
    data = {
        "Model": model_name,
        "Total Parameters": f"{params['total']:,}",
        "Model Size (FP32)": f"{params['total'] * 4 / 1024 / 1024:.1f} MB",
        "Training Epochs": len(history["train_loss"]),
        "Best Epoch": history.get("best_epoch", "N/A"),
        "Best Valid Dice": f"{history.get('best_dice', 0):.4f}",
        "Test Mean Dice": f"{test_metrics['mean_dice']:.4f}",
        "Test Mean IoU": f"{test_metrics['mean_iou']:.4f}",
        "Test Pixel Accuracy": f"{test_metrics['pixel_accuracy']:.4f}",
    }
    return pd.DataFrame([data]).T.rename(columns={0: "Value"})


def plot_threshold_analysis(model, loader, device, save_path=None, num_classes=NUM_CLASSES):
    import torch
    model.eval()
    thresholds = np.arange(0.1, 0.95, 0.05)
    dice_curves = np.zeros((num_classes, len(thresholds)))
    class_tp = np.zeros((num_classes, len(thresholds)))
    class_pred_sum = np.zeros((num_classes, len(thresholds)))
    class_target_sum = np.zeros(num_classes)

    with torch.no_grad():
        for imgs, masks in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            masks_np = masks.numpy()
            for c in range(num_classes):
                target_c = (masks_np == c) & (masks_np != IGNORE_LABEL)
                class_target_sum[c] += target_c.sum()
                for ti, t in enumerate(thresholds):
                    pred_c = probs[:, c] >= t
                    class_tp[c, ti] += (pred_c & target_c).sum()
                    class_pred_sum[c, ti] += pred_c.sum()

    opt = {}
    for c in range(num_classes):
        for ti in range(len(thresholds)):
            inter = class_tp[c, ti]
            union = class_pred_sum[c, ti] + class_target_sum[c]
            dice_curves[c, ti] = (2 * inter + 1e-6) / (union + 1e-6)
        best_idx = np.argmax(dice_curves[c])
        opt[CLASS_NAMES[c]] = {"threshold": round(float(thresholds[best_idx]), 2),
                                "dice": round(float(dice_curves[c, best_idx]), 4)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for c in range(num_classes):
        axes[0].plot(thresholds, dice_curves[c], label=CLASS_NAMES[c], color=PALETTE[c], linewidth=2)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Dice Score")
    axes[0].set_title("Dice vs Threshold")
    axes[0].legend(frameon=True)
    axes[0].grid(alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    x = np.arange(num_classes)
    ts = [opt[n]["threshold"] for n in CLASS_NAMES]
    ds = [opt[n]["dice"] for n in CLASS_NAMES]
    bars = axes[1].bar(x, ts, color=[PALETTE[i] for i in range(num_classes)], edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylabel("Optimal Threshold")
    axes[1].set_title("Best Threshold per Class")
    axes[1].set_ylim(0, 1)
    for bar, d in zip(bars, ds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"D={d:.3f}", ha="center", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("Per-Class Threshold Optimization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, save_path)
    return opt


def plot_roc_curves(model, loader, device, save_path=None, num_classes=NUM_CLASSES):
    from sklearn.metrics import roc_curve, auc
    import torch
    model.eval()

    all_probs = {c: [] for c in range(num_classes)}
    all_labels = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for imgs, masks in loader:
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            masks_np = masks.numpy()
            valid = masks_np != IGNORE_LABEL
            for c in range(num_classes):
                flat_valid = valid.flatten()
                all_probs[c].extend(probs[:, c].flatten()[flat_valid].tolist())
                all_labels[c].extend(((masks_np == c) & valid).flatten()[flat_valid].astype(int).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    aucs = {}
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels[c], all_probs[c])
        roc_auc = auc(fpr, tpr)
        aucs[CLASS_NAMES[c]] = round(roc_auc, 4)
        axes[0].plot(fpr, tpr, color=PALETTE[c], linewidth=2,
                     label=f"{CLASS_NAMES[c]} (AUC={roc_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves (One-vs-Rest)")
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    x = np.arange(num_classes)
    bars = axes[1].bar(x, [aucs[n] for n in CLASS_NAMES],
                       color=[PALETTE[i] for i in range(num_classes)], edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC per Class")
    axes[1].set_ylim(0, 1)
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{bar.get_height():.3f}", ha="center", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    mean_auc = np.mean(list(aucs.values()))
    plt.suptitle(f"ROC Analysis (Mean AUC: {mean_auc:.3f})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _finish(fig, save_path)
    return aucs
