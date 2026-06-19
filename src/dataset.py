import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import NUM_CLASSES


class BCSSDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.img_dir = Path(data_dir) / split / "images"
        self.mask_dir = Path(data_dir) / split / "masks"
        self.filenames = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        image = np.array(Image.open(self.img_dir / fname))
        mask = np.array(Image.open(self.mask_dir / fname))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        return image, mask


def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
                 border_mode=0, p=0.5),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def compute_class_weights(data_dir, split="train"):
    mask_dir = Path(data_dir) / split / "masks"
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for fname in sorted(os.listdir(mask_dir)):
        mask = np.array(Image.open(mask_dir / fname))
        for c in range(NUM_CLASSES):
            pixel_counts[c] += np.sum(mask == c)
    total = pixel_counts.sum()
    weights = total / (NUM_CLASSES * pixel_counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    return weights.astype(np.float32)
