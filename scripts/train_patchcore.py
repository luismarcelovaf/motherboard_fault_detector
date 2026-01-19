#!/usr/bin/env python
"""
Train PatchCore anomaly detection model.
Builds memory bank from normal motherboard images.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from tqdm import tqdm

from src.models.patchcore import PatchCoreModel, create_patchcore_from_config
from src.data.dataset import create_patchcore_dataloader
from src.data.augmentation import PatchCoreAugmentation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PatchCore anomaly detection model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/normal",
        help="Directory containing normal images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/models/patchcore.pt",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=50,
        help="Number of augmented versions per image",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("PatchCore Training Script")
    print("=" * 60)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {}

    # Override config with command line args
    if args.device:
        config.setdefault("hardware", {})["device"] = args.device

    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nPlease organize your data as follows:")
        print(f"  {data_dir}/")
        print("    ├── normal_image_1.jpg")
        print("    ├── normal_image_2.jpg")
        print("    └── ...")
        return 1

    # Count images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [
        f for f in data_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    print(f"\nFound {len(image_files)} normal images in {data_dir}")

    if len(image_files) == 0:
        print("Error: No images found!")
        return 1

    # Create model
    print("\nInitializing PatchCore model...")
    device = config.get("hardware", {}).get("device", args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    model = create_patchcore_from_config(config)
    model.device = device
    model.feature_extractor = model.feature_extractor.to(device)

    print(f"  Backbone: {model.backbone_name}")
    print(f"  Feature layers: {model.layers_to_extract}")
    print(f"  Input size: {model.input_size}")
    print(f"  Coreset sampling ratio: {model.coreset_sampling_ratio}")
    print(f"  Device: {device}")

    # Create augmentation pipeline
    print("\nSetting up augmentation pipeline...")
    patchcore_config = config.get("patchcore", {})
    augmentation = PatchCoreAugmentation(
        target_size=tuple(patchcore_config.get("input_size", [256, 256]))
    )

    # Create data loader
    print(f"\nCreating data loader (augmentation factor: {args.augmentation_factor})...")
    dataloader = create_patchcore_dataloader(
        normal_dir=data_dir,
        augmentation=augmentation,
        augmentation_factor=args.augmentation_factor,
        batch_size=args.batch_size,
        num_workers=config.get("hardware", {}).get("num_workers", 4),
    )

    total_samples = len(image_files) * args.augmentation_factor
    print(f"  Total samples (with augmentation): {total_samples}")

    # Build memory bank
    print("\nBuilding memory bank...")
    print("This may take a while depending on the number of images...")

    model.fit(dataloader)

    print(f"\nMemory bank built successfully!")
    print(f"  Memory bank size: {len(model.memory_bank)}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {output_path}")

    # Print usage instructions
    print("\nNext steps:")
    print("1. Train the classifier:")
    print(f"   python scripts/train_classifier.py --config {args.config}")
    print("\n2. Run inference:")
    print(f"   python scripts/inference.py --image <path_to_image>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
