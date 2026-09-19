import os
import glob
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: int = 256, is_train: bool = True) -> A.Compose:
    if is_train:
        return A.Compose(
            [
                A.RandomCrop(width=img_size, height=img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            additional_targets={'image_b': 'image', 'mask': 'mask'}
        )
    else:
        return A.Compose(
            [
                A.CenterCrop(width=img_size, height=img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            additional_targets={'image_b': 'image', 'mask': 'mask'}
        )

class LEVIRCDDataset(Dataset):
    """
    Dataset loader for LEVIR-CD / LEVIR-CD+ bitemporal satellite image pairs.
    Folder Structure:
      root_dir/
        ├── A/       (Pre-change RGB images)
        ├── B/       (Post-change RGB images)
        └── label/   (Binary change mask: 0=No Change, 255=Change)
    """
    def __init__(self, root_dir: str, img_size: int = 256, is_train: bool = True):
        self.root_dir = root_dir
        self.img_a_dir = os.path.join(root_dir, 'A')
        self.img_b_dir = os.path.join(root_dir, 'B')
        self.label_dir = os.path.join(root_dir, 'label')

        self.filenames = sorted([
            f for f in os.listdir(self.img_a_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        
        self.transform = get_transforms(img_size=img_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filename = self.filenames[idx]
        
        path_a = os.path.join(self.img_a_dir, filename)
        path_b = os.path.join(self.img_b_dir, filename)
        path_label = os.path.join(self.label_dir, filename)

        img_a = np.array(Image.open(path_a).convert("RGB"))
        img_b = np.array(Image.open(path_b).convert("RGB"))
        
        if os.path.exists(path_label):
            mask = np.array(Image.open(path_label).convert("L"))
            # Normalize mask to binary 0 or 1
            mask = (mask > 128).astype(np.float32)
        else:
            mask = np.zeros((img_a.shape[0], img_a.shape[1]), dtype=np.float32)

        augmented = self.transform(image=img_a, image_b=img_b, mask=mask)
        
        return {
            'img_a': augmented['image'],           # Shape: [3, H, W]
            'img_b': augmented['image_b'],         # Shape: [3, H, W]
            'mask': augmented['mask'].unsqueeze(0), # Shape: [1, H, W]
            'filename': filename
        }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates optimized PyTorch DataLoaders for Train, Val, and Test splits.
    """
    train_dataset = LEVIRCDDataset(os.path.join(data_dir, 'train'), img_size=img_size, is_train=True)
    val_dataset = LEVIRCDDataset(os.path.join(data_dir, 'val'), img_size=img_size, is_train=False)
    test_dataset = LEVIRCDDataset(os.path.join(data_dir, 'test'), img_size=img_size, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader