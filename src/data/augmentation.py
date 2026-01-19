"""
Data augmentation pipeline for motherboard fault detection.
Uses Albumentations for robust image transforms to expand small datasets.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import random


class AugmentationPipeline:
    """
    Augmentation pipeline for expanding small datasets.
    Designed to create 50x augmented versions of each image.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        # Geometric transforms
        rotation_limit: int = 30,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        shift_limit: float = 0.1,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.3,
        perspective_prob: float = 0.3,
        elastic_prob: float = 0.2,
        # Color transforms
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        hue_shift_limit: int = 10,
        saturation_limit: int = 20,
        gamma_range: Tuple[int, int] = (80, 120),
        # Noise and blur
        gaussian_noise_var_limit: Tuple[int, int] = (10, 50),
        gaussian_blur_limit: Tuple[int, int] = (3, 7),
        motion_blur_limit: int = 7,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            target_size: Output image size (height, width)
            rotation_limit: Max rotation angle in degrees
            scale_range: Min and max scale factors
            shift_limit: Max shift as fraction of image size
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            perspective_prob: Probability of perspective transform
            elastic_prob: Probability of elastic transform
            brightness_limit: Max brightness adjustment
            contrast_limit: Max contrast adjustment
            hue_shift_limit: Max hue shift
            saturation_limit: Max saturation adjustment
            gamma_range: Gamma correction range (percentage)
            gaussian_noise_var_limit: Gaussian noise variance range
            gaussian_blur_limit: Gaussian blur kernel size range
            motion_blur_limit: Motion blur kernel size
        """
        self.target_size = target_size

        # Build the training augmentation pipeline
        self.train_transform = A.Compose([
            # Resize first
            A.Resize(height=target_size[0], width=target_size[1]),

            # Geometric transforms
            A.Affine(
                translate_percent={"x": (-shift_limit, shift_limit), "y": (-shift_limit, shift_limit)},
                scale=(scale_range[0], scale_range[1]),
                rotate=(-rotation_limit, rotation_limit),
                mode=cv2.BORDER_REFLECT_101,
                p=0.8
            ),
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.VerticalFlip(p=vertical_flip_prob),
            A.Perspective(scale=(0.02, 0.05), p=perspective_prob),
            A.ElasticTransform(
                alpha=50,
                sigma=10,
                p=elastic_prob
            ),

            # Color transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=1.0
                ),
                A.RandomGamma(
                    gamma_limit=gamma_range,
                    p=1.0
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.7),

            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=hue_shift_limit,
                    sat_shift_limit=saturation_limit,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
                A.ChannelShuffle(p=1.0),
            ], p=0.5),

            # Noise and blur
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.4),

            A.OneOf([
                A.GaussianBlur(blur_limit=gaussian_blur_limit, p=1.0),
                A.MotionBlur(blur_limit=motion_blur_limit, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.3),

            # Additional augmentations for robustness
            A.OneOf([
                A.ImageCompression(quality_range=(70, 100), p=1.0),
                A.Downscale(scale_range=(0.5, 0.9), p=1.0),
            ], p=0.2),

            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(int(target_size[0] * 0.05), int(target_size[0] * 0.1)),
                hole_width_range=(int(target_size[1] * 0.05), int(target_size[1] * 0.1)),
                p=0.2
            ),
        ])

        # Validation/test transform (only resize and normalize)
        self.val_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
        ])

        # PyTorch transform (includes normalization and tensor conversion)
        self.to_tensor_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],   # ImageNet std
            ),
            ToTensorV2(),
        ])

    def augment(self, image: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Apply augmentation to a single image.

        Args:
            image: Input RGB image as numpy array
            is_training: Whether to apply training augmentations

        Returns:
            Augmented image as numpy array
        """
        if is_training:
            transformed = self.train_transform(image=image)
        else:
            transformed = self.val_transform(image=image)

        return transformed["image"]

    def augment_to_tensor(
        self,
        image: np.ndarray,
        is_training: bool = True
    ) -> "torch.Tensor":
        """
        Apply augmentation and convert to PyTorch tensor.

        Args:
            image: Input RGB image as numpy array
            is_training: Whether to apply training augmentations

        Returns:
            Augmented image as PyTorch tensor
        """
        # Apply augmentations
        augmented = self.augment(image, is_training)

        # Convert to tensor with normalization
        transformed = self.to_tensor_transform(image=augmented)

        return transformed["image"]

    def generate_augmented_batch(
        self,
        image: np.ndarray,
        num_augmentations: int = 50
    ) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of a single image.

        Args:
            image: Input RGB image
            num_augmentations: Number of augmented versions to generate

        Returns:
            List of augmented images
        """
        augmented_images = []

        for _ in range(num_augmentations):
            aug_image = self.augment(image, is_training=True)
            augmented_images.append(aug_image)

        return augmented_images

    def save_augmented_images(
        self,
        image: np.ndarray,
        output_dir: Path,
        base_name: str,
        num_augmentations: int = 50,
        save_original: bool = True
    ) -> List[Path]:
        """
        Generate and save augmented images to disk.

        Args:
            image: Input RGB image
            output_dir: Directory to save augmented images
            base_name: Base filename for saved images
            num_augmentations: Number of augmented versions to generate
            save_original: Whether to save the original (resized) image

        Returns:
            List of paths to saved images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        # Save original (resized)
        if save_original:
            original_resized = self.val_transform(image=image)["image"]
            original_path = output_dir / f"{base_name}_original.png"
            # Convert RGB to BGR for OpenCV
            cv2.imwrite(str(original_path), cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR))
            saved_paths.append(original_path)

        # Generate and save augmented versions
        for i in range(num_augmentations):
            aug_image = self.augment(image, is_training=True)
            aug_path = output_dir / f"{base_name}_aug_{i:03d}.png"
            cv2.imwrite(str(aug_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            saved_paths.append(aug_path)

        return saved_paths


class PatchCoreAugmentation:
    """
    Lighter augmentation pipeline specifically for PatchCore memory bank building.
    Uses more conservative transforms to maintain feature consistency.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
    ):
        """
        Initialize PatchCore augmentation pipeline.

        Args:
            target_size: Output image size (height, width)
        """
        self.target_size = target_size

        # Conservative augmentations for memory bank
        self.transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),

            # Light geometric transforms
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),

            # Light color transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),

            # Very light noise
            A.GaussNoise(var_limit=(5, 20), p=0.2),
        ])

        # Normalization for model input
        self.normalize = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image."""
        return self.transform(image=image)["image"]

    def to_tensor(self, image: np.ndarray) -> "torch.Tensor":
        """Convert image to normalized tensor."""
        return self.normalize(image=image)["image"]

    def generate_augmented_batch(
        self,
        image: np.ndarray,
        num_augmentations: int = 50
    ) -> List[np.ndarray]:
        """Generate multiple augmented versions for memory bank."""
        augmented_images = []

        # Add resized original
        resized = A.Resize(
            height=self.target_size[0],
            width=self.target_size[1]
        )(image=image)["image"]
        augmented_images.append(resized)

        # Generate augmented versions
        for _ in range(num_augmentations - 1):
            aug_image = self.augment(image)
            augmented_images.append(aug_image)

        return augmented_images


def create_augmentation_pipeline_from_config(config: dict) -> AugmentationPipeline:
    """
    Create an AugmentationPipeline from a configuration dictionary.

    Args:
        config: Configuration dictionary with augmentation parameters

    Returns:
        Configured AugmentationPipeline instance
    """
    preproc_config = config.get("preprocessing", {})
    aug_config = config.get("augmentation", {})

    geo_config = aug_config.get("geometric", {})
    color_config = aug_config.get("color", {})
    noise_config = aug_config.get("noise", {})

    return AugmentationPipeline(
        target_size=tuple(preproc_config.get("target_size", [224, 224])),
        rotation_limit=geo_config.get("rotation_limit", 30),
        scale_range=tuple(geo_config.get("scale_range", [0.8, 1.2])),
        shift_limit=geo_config.get("shift_limit", 0.1),
        horizontal_flip_prob=geo_config.get("horizontal_flip_prob", 0.5),
        vertical_flip_prob=geo_config.get("vertical_flip_prob", 0.3),
        perspective_prob=geo_config.get("perspective_prob", 0.3),
        elastic_prob=geo_config.get("elastic_prob", 0.2),
        brightness_limit=color_config.get("brightness_limit", 0.2),
        contrast_limit=color_config.get("contrast_limit", 0.2),
        hue_shift_limit=color_config.get("hue_shift_limit", 10),
        saturation_limit=color_config.get("saturation_limit", 20),
        gamma_range=tuple(color_config.get("gamma_range", [80, 120])),
        gaussian_noise_var_limit=tuple(noise_config.get("gaussian_noise_var_limit", [10, 50])),
        gaussian_blur_limit=tuple(noise_config.get("gaussian_blur_limit", [3, 7])),
        motion_blur_limit=noise_config.get("motion_blur_limit", 7),
    )


def create_patchcore_augmentation_from_config(config: dict) -> PatchCoreAugmentation:
    """
    Create a PatchCoreAugmentation from a configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured PatchCoreAugmentation instance
    """
    patchcore_config = config.get("patchcore", {})
    input_size = patchcore_config.get("input_size", [256, 256])

    return PatchCoreAugmentation(target_size=tuple(input_size))
