#!/usr/bin/env python
"""
Train EfficientNet classifier for defect classification.
Uses transfer learning with heavy augmentation for small datasets.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models.classifier import (
    DefectClassifier,
    ClassifierTrainer,
    create_classifier_from_config,
    create_trainer_from_config,
)
from src.data.dataset import (
    MotherboardDataset,
    create_data_loaders,
    DEFECT_CLASSES,
    CLASS_NAMES,
)
from src.data.augmentation import (
    AugmentationPipeline,
    create_augmentation_pipeline_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train defect classifier using transfer learning"
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
        default="data/raw",
        help="Root directory containing defect class subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/classifier.pt",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=50,
        help="Number of augmented versions per image",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (0 to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    return parser.parse_args()


def count_samples(data_dir: Path) -> dict:
    """Count samples per class."""
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    counts = {}

    for class_name in DEFECT_CLASSES.keys():
        class_dir = data_dir / class_name
        if class_dir.exists():
            count = len([
                f for f in class_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ])
            counts[class_name] = count
        else:
            counts[class_name] = 0

    return counts


def train_single_fold(
    train_samples,
    val_samples,
    augmentation_pipeline,
    config,
    args,
    fold=None,
):
    """Train a single fold."""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Create datasets
    from torch.utils.data import DataLoader

    # Training dataset with augmentation
    train_dataset = MotherboardDataset(
        data_dir=args.data_dir,
        augmentation_pipeline=augmentation_pipeline,
        is_training=True,
        augmentation_factor=args.augmentation_factor,
    )
    train_dataset.samples = train_samples

    # Validation dataset without augmentation
    val_dataset = MotherboardDataset(
        data_dir=args.data_dir,
        augmentation_pipeline=augmentation_pipeline,
        is_training=False,
        augmentation_factor=1,
    )
    val_dataset.samples = val_samples

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.get("hardware", {}).get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get("hardware", {}).get("num_workers", 4),
        pin_memory=True,
    )

    # Create model
    model = create_classifier_from_config(config)

    # Create trainer
    trainer = create_trainer_from_config(model, config)
    trainer.device = device
    trainer.model = trainer.model.to(device)

    # Get class weights for imbalanced data
    class_weights = train_dataset.get_class_weights()

    if fold is not None:
        print(f"\n--- Fold {fold + 1} ---")

    print(f"Training samples: {len(train_samples)} (x{args.augmentation_factor} = {len(train_samples) * args.augmentation_factor})")
    print(f"Validation samples: {len(val_samples)}")

    # Train
    training_config = config.get("classifier", {}).get("training", {})
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        class_weights=class_weights,
    )

    return trainer, result


def main():
    args = parse_args()

    print("=" * 60)
    print("Defect Classifier Training Script")
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

    # Check data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nPlease organize your data as follows:")
        print(f"  {data_dir}/")
        print("    ├── burn_marks/")
        print("    │   ├── image1.jpg")
        print("    │   └── ...")
        print("    ├── reuse_marks/")
        print("    ├── liquid_damage/")
        print("    ├── label_tampering/")
        print("    └── other_tampering/")
        return 1

    # Count samples
    print("\nDataset Summary:")
    sample_counts = count_samples(data_dir)
    total_samples = 0
    for class_name, count in sample_counts.items():
        print(f"  {class_name}: {count} images")
        total_samples += count

    print(f"  Total: {total_samples} images")

    if total_samples == 0:
        print("\nError: No images found!")
        return 1

    # Create augmentation pipeline
    print("\nSetting up augmentation pipeline...")
    augmentation_pipeline = create_augmentation_pipeline_from_config(config)
    print(f"  Augmentation factor: {args.augmentation_factor}")
    print(f"  Effective training size: ~{total_samples * args.augmentation_factor}")

    # Load all samples
    full_dataset = MotherboardDataset(
        data_dir=data_dir,
        augmentation_pipeline=augmentation_pipeline,
        is_training=False,
        augmentation_factor=1,
    )

    samples = full_dataset.samples
    labels = [label for _, label in samples]

    # Check minimum samples per class for cross-validation
    from collections import Counter
    label_counts = Counter(labels)
    min_class_count = min(label_counts.values())

    # Cross-validation or single split
    can_do_cv = args.cv_folds > 1 and min_class_count >= args.cv_folds

    if can_do_cv:
        print(f"\nPerforming {args.cv_folds}-fold cross-validation...")

        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            trainer, result = train_single_fold(
                train_samples=train_samples,
                val_samples=val_samples,
                augmentation_pipeline=augmentation_pipeline,
                config=config,
                args=args,
                fold=fold,
            )
            fold_results.append(result)

        # Summary
        best_f1s = [r["best_f1"] for r in fold_results]
        print("\n" + "=" * 60)
        print("Cross-Validation Results")
        print("=" * 60)
        print(f"Best F1 scores: {[f'{f:.4f}' for f in best_f1s]}")
        print(f"Mean F1: {np.mean(best_f1s):.4f} (+/- {np.std(best_f1s):.4f})")

        # Train final model on all data
        print("\nTraining final model on all data...")
        val_size = max(1, int(len(samples) * args.val_split))
        train_samples = samples[val_size:]
        val_samples = samples[:val_size]

        trainer, result = train_single_fold(
            train_samples=train_samples,
            val_samples=val_samples,
            augmentation_pipeline=augmentation_pipeline,
            config=config,
            args=args,
        )

    else:
        # Single train/val split (either requested or too few samples for CV)
        if args.cv_folds > 1 and not can_do_cv:
            print(f"\n[!] Too few samples per class for {args.cv_folds}-fold CV (min class has {min_class_count} samples)")
        print(f"Using single train/val split ({1-args.val_split:.0%}/{args.val_split:.0%})...")

        # Shuffle and split
        import random
        random.seed(42)
        indices = list(range(len(samples)))
        random.shuffle(indices)

        val_size = int(len(samples) * args.val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_samples = [samples[i] for i in train_indices]
        val_samples = [samples[i] for i in val_indices]

        trainer, result = train_single_fold(
            train_samples=train_samples,
            val_samples=val_samples,
            augmentation_pipeline=augmentation_pipeline,
            config=config,
            args=args,
        )

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"Model saved to: {output_path}")

    # Print usage instructions
    print("\nNext steps:")
    print("1. Run inference:")
    print(f"   python scripts/inference.py --image <path_to_image>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
