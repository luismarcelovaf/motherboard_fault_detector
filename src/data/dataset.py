"""
PyTorch Dataset classes for motherboard fault detection.
Includes datasets for both classifier training and PatchCore memory bank building.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import random

from .augmentation import AugmentationPipeline, PatchCoreAugmentation
from .preprocessing import ImagePreprocessor


# Valid image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def discover_defect_classes(data_dir: Union[str, Path]) -> Tuple[Dict[str, int], List[str]]:
    """
    Discover classes dynamically from subdirectories.

    Only includes folders that contain at least one image file.
    Includes 'normal' as first class (index 0).

    Args:
        data_dir: Root directory containing class subdirectories

    Returns:
        Tuple of (class_to_idx dict, list of class names)
    """
    data_dir = Path(data_dir)
    classes = {}
    idx = 0

    if not data_dir.exists():
        return {}, []

    # Sort with 'normal' first
    subdirs = sorted(data_dir.iterdir(), key=lambda x: (x.name != "normal", x.name))

    for subdir in subdirs:
        if subdir.is_dir():
            # Check if folder has any images
            has_images = any(
                list(subdir.glob(f"*{ext}")) + list(subdir.glob(f"*{ext.upper()}"))
                for ext in IMAGE_EXTENSIONS
            )
            if has_images:
                classes[subdir.name] = idx
                idx += 1

    class_names = list(classes.keys())
    return classes, class_names


# For backward compatibility - these will be None until a dataset is created
# Prefer using dataset.defect_classes and dataset.class_names instead
DEFECT_CLASSES: Optional[Dict[str, int]] = None
CLASS_NAMES: Optional[List[str]] = None


class MotherboardDataset(Dataset):
    """
    PyTorch Dataset for motherboard classification.
    Supports on-the-fly augmentation for training.

    Classes are discovered dynamically from subdirectory names.
    Includes 'normal' as a class.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        augmentation_pipeline: Optional[AugmentationPipeline] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        is_training: bool = True,
        augmentation_factor: int = 1,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Root directory containing class subdirectories
            augmentation_pipeline: Augmentation pipeline for training
            preprocessor: Image preprocessor (optional, augmentation handles resize)
            is_training: Whether this is for training (applies augmentation)
            augmentation_factor: Number of augmented versions per image
        """
        self.data_dir = Path(data_dir)
        self.augmentation = augmentation_pipeline
        self.preprocessor = preprocessor
        self.is_training = is_training
        self.augmentation_factor = augmentation_factor if is_training else 1

        # Discover defect classes dynamically (excludes normal)
        self.defect_classes, self.class_names = discover_defect_classes(self.data_dir)
        self.num_classes = len(self.defect_classes)

        # Collect image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        """Load image paths and labels from directory structure."""
        # Load defect images only (normal excluded)
        for class_name, class_idx in self.defect_classes.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((img_path, class_idx))

        print(f"Loaded {len(self.samples)} images from {self.data_dir}")
        print(f"Discovered {self.num_classes} classes: {self.class_names}")

    def __len__(self) -> int:
        return len(self.samples) * self.augmentation_factor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor, class label)
        """
        # Map augmented index to original sample
        original_idx = idx // self.augmentation_factor
        img_path, label = self.samples[original_idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply preprocessing if available
        if self.preprocessor is not None:
            image = self.preprocessor.preprocess_for_model(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                normalize=False
            )

        # Apply augmentation and convert to tensor
        if self.augmentation is not None:
            image_tensor = self.augmentation.augment_to_tensor(
                image, is_training=self.is_training
            )
        else:
            # Manual conversion if no augmentation pipeline
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        return image_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset.

        Returns:
            Tensor of class weights inversely proportional to class frequency
        """
        # Count samples per class
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        # Calculate weights using dynamically discovered num_classes
        total = sum(class_counts.values())
        weights = torch.zeros(self.num_classes)

        for class_idx in range(self.num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            weights[class_idx] = total / (self.num_classes * count)

        return weights


class PatchCoreDataset(Dataset):
    """
    Dataset for PatchCore memory bank building.
    Only uses normal images with conservative augmentation.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        augmentation: Optional[PatchCoreAugmentation] = None,
        augmentation_factor: int = 50,
    ):
        """
        Initialize the PatchCore dataset.

        Args:
            data_dir: Directory containing normal images
            augmentation: PatchCore augmentation pipeline
            augmentation_factor: Number of augmented versions per image
        """
        self.data_dir = Path(data_dir)
        self.augmentation = augmentation or PatchCoreAugmentation()
        self.augmentation_factor = augmentation_factor

        # Collect normal image paths
        self.image_paths: List[Path] = []
        self._load_images()

    def _load_images(self):
        """Load normal image paths."""
        for img_path in self.data_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                self.image_paths.append(img_path)

        print(f"Loaded {len(self.image_paths)} normal images for PatchCore")

    def __len__(self) -> int:
        return len(self.image_paths) * self.augmentation_factor

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Image tensor
        """
        # Map augmented index to original image
        original_idx = idx // self.augmentation_factor
        img_path = self.image_paths[original_idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation
        augmented = self.augmentation.augment(image)

        # Convert to tensor with normalization
        tensor = self.augmentation.to_tensor(augmented)

        return tensor


class AnomalyDetectionDataset(Dataset):
    """
    Dataset for anomaly detection inference.
    Returns both normal and defect images with their labels.

    Classes are discovered dynamically from subdirectory names.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        target_size: Tuple[int, int] = (256, 256),
    ):
        """
        Initialize the anomaly detection dataset.

        Args:
            data_dir: Root directory containing 'normal' and defect subdirectories
            target_size: Output image size
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size

        # Discover classes dynamically from folder structure
        self.defect_classes, self.class_names = discover_defect_classes(self.data_dir)

        # Collect all images with anomaly labels
        self.samples: List[Tuple[Path, bool, str]] = []  # (path, is_anomaly, class_name)
        self._load_samples()

        # Transform for inference
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        self.transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _load_samples(self):
        """Load all image paths with anomaly labels."""
        # Load normal images
        normal_dir = self.data_dir / "normal"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*"):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, False, "normal"))

        # Load defect images using dynamically discovered classes
        for class_name in self.defect_classes.keys():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((img_path, True, class_name))

        print(f"Loaded {len(self.samples)} images for anomaly detection")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, bool, str, Path]:
        """
        Get a single sample.

        Returns:
            Tuple of (image tensor, is_anomaly, class_name, image_path)
        """
        img_path, is_anomaly, class_name = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transform
        transformed = self.transform(image=image)
        tensor = transformed["image"]

        return tensor, is_anomaly, class_name, img_path


def create_data_loaders(
    data_dir: Union[str, Path],
    augmentation_pipeline: AugmentationPipeline,
    batch_size: int = 16,
    augmentation_factor: int = 50,
    num_workers: int = 4,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        data_dir: Root directory containing class subdirectories
        augmentation_pipeline: Augmentation pipeline for training
        batch_size: Batch size for data loaders
        augmentation_factor: Number of augmented versions per image
        num_workers: Number of worker processes
        val_split: Fraction of data for validation
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = MotherboardDataset(
        data_dir=data_dir,
        augmentation_pipeline=augmentation_pipeline,
        is_training=True,
        augmentation_factor=1,  # We'll handle augmentation separately for split
    )

    # Split into train and validation
    num_samples = len(full_dataset.samples)
    indices = list(range(num_samples))
    random.seed(random_seed)
    random.shuffle(indices)

    split_idx = int(num_samples * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create separate datasets
    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples = [full_dataset.samples[i] for i in val_indices]

    # Create training dataset with augmentation
    train_dataset = MotherboardDataset(
        data_dir=data_dir,
        augmentation_pipeline=augmentation_pipeline,
        is_training=True,
        augmentation_factor=augmentation_factor,
    )
    train_dataset.samples = train_samples

    # Create validation dataset without augmentation
    val_dataset = MotherboardDataset(
        data_dir=data_dir,
        augmentation_pipeline=augmentation_pipeline,
        is_training=False,
        augmentation_factor=1,
    )
    val_dataset.samples = val_samples

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_patchcore_dataloader(
    normal_dir: Union[str, Path],
    augmentation: Optional[PatchCoreAugmentation] = None,
    augmentation_factor: int = 50,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create data loader for PatchCore memory bank building.

    Args:
        normal_dir: Directory containing normal images
        augmentation: PatchCore augmentation pipeline
        augmentation_factor: Number of augmented versions per image
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        DataLoader for normal images
    """
    dataset = PatchCoreDataset(
        data_dir=normal_dir,
        augmentation=augmentation,
        augmentation_factor=augmentation_factor,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Order doesn't matter for memory bank
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
