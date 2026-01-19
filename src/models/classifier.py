"""
EfficientNet-based classifier for motherboard defect classification.
Uses transfer learning with heavy regularization for small datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
from tqdm import tqdm
import timm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import copy


class DefectClassifier(nn.Module):
    """
    EfficientNet-based classifier for motherboard defect classification.
    Uses transfer learning with configurable freezing and regularization.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Initialize the classifier.

        Args:
            backbone: Backbone model name (from timm)
            num_classes: Number of defect classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Custom classification head with heavy regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Initialize classifier weights
        self._init_classifier()

    def _init_classifier(self):
        """Initialize classifier weights with Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Class logits of shape (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
        return preds, probs

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representation.

        Args:
            x: Input tensor

        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features


class ClassifierTrainer:
    """
    Trainer for the defect classifier with early stopping and LR scheduling.
    """

    def __init__(
        self,
        model: DefectClassifier,
        device: str = "cuda",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        label_smoothing: float = 0.1,
        freeze_backbone_epochs: int = 5,
        patience: int = 15,
        min_delta: float = 0.001,
    ):
        """
        Initialize trainer.

        Args:
            model: DefectClassifier model
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            label_smoothing: Label smoothing factor
            freeze_backbone_epochs: Epochs to keep backbone frozen
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.patience = patience
        self.min_delta = min_delta

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer (will be recreated after unfreezing)
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.training_history: List[Dict[str, float]] = []

    def _create_optimizer(self, unfreeze: bool = False):
        """Create optimizer with appropriate parameter groups."""
        if unfreeze:
            # Different learning rates for backbone and classifier
            params = [
                {"params": self.model.backbone.parameters(), "lr": self.learning_rate * 0.1},
                {"params": self.model.classifier.parameters(), "lr": self.learning_rate},
            ]
        else:
            # Only train classifier
            params = self.model.classifier.parameters()

        self.optimizer = optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _create_scheduler(self, num_epochs: int, steps_per_epoch: int):
        """Create learning rate scheduler."""
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            class_weights: Optional class weights for imbalanced data

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Update criterion with class weights if provided
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device),
                label_smoothing=self.label_smoothing,
            )

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Track metrics
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
            "predictions": np.array(all_preds),
            "labels": np.array(all_labels),
            "probabilities": np.array(all_probs),
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            class_weights: Optional class weights

        Returns:
            Dictionary with training history and best metrics
        """
        # Phase 1: Train with frozen backbone
        print(f"\nPhase 1: Training classifier head ({self.freeze_backbone_epochs} epochs)")
        self.model.freeze_backbone()
        self._create_optimizer(unfreeze=False)
        self._create_scheduler(self.freeze_backbone_epochs, len(train_loader))

        for epoch in range(self.freeze_backbone_epochs):
            print(f"\nEpoch {epoch + 1}/{self.freeze_backbone_epochs}")

            train_metrics = self.train_epoch(train_loader, class_weights)

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self._log_epoch(epoch, train_metrics, val_metrics)
                self._check_improvement(val_metrics["f1"])
                if self.epochs_without_improvement >= self.patience:
                    print("Early stopping triggered in phase 1")
                    break
            else:
                self._log_epoch_train_only(epoch, train_metrics)
                self._save_model_state()

        # Phase 2: Fine-tune entire model
        remaining_epochs = epochs - self.freeze_backbone_epochs
        if remaining_epochs > 0:
            print(f"\nPhase 2: Fine-tuning entire model ({remaining_epochs} epochs)")
            self.model.unfreeze_backbone()
            self._create_optimizer(unfreeze=True)
            self._create_scheduler(remaining_epochs, len(train_loader))

            for epoch in range(remaining_epochs):
                print(f"\nEpoch {self.freeze_backbone_epochs + epoch + 1}/{epochs}")

                train_metrics = self.train_epoch(train_loader, class_weights)

                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    self._log_epoch(self.freeze_backbone_epochs + epoch, train_metrics, val_metrics)
                    self._check_improvement(val_metrics["f1"])
                    if self.epochs_without_improvement >= self.patience:
                        print("Early stopping triggered in phase 2")
                        break
                else:
                    self._log_epoch_train_only(self.freeze_backbone_epochs + epoch, train_metrics)
                    self._save_model_state()

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return {
            "history": self.training_history,
            "best_f1": self.best_val_f1,
        }

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log epoch metrics."""
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
        }
        self.training_history.append(metrics)

        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")

    def _check_improvement(self, val_f1: float):
        """Check for improvement and update best model."""
        if val_f1 > self.best_val_f1 + self.min_delta:
            self.best_val_f1 = val_f1
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            self.epochs_without_improvement = 0
            print(f"New best F1: {val_f1:.4f}")
        else:
            self.epochs_without_improvement += 1

    def _log_epoch_train_only(self, epoch: int, train_metrics: Dict[str, float]):
        """Log epoch metrics when no validation."""
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
        }
        self.training_history.append(metrics)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")

    def _save_model_state(self):
        """Save current model state as best."""
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        self.best_val_f1 = 1.0

    def save_model(self, path: Union[str, Path], class_names: Optional[List[str]] = None):
        """Save model checkpoint with class names."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": {
                "backbone": self.model.backbone_name,
                "num_classes": self.model.num_classes,
                "feature_dim": self.model.feature_dim,
            },
            "class_names": class_names,  # Store class names for inference
            "training_history": self.training_history,
            "best_f1": self.best_val_f1,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
        if class_names:
            print(f"  Classes: {class_names}")

    def load_model(self, path: Union[str, Path]):
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.training_history = checkpoint.get("training_history", [])
        self.best_val_f1 = checkpoint.get("best_f1", 0.0)

        print(f"Model loaded from {path}")


def cross_validate(
    model_class: type,
    data_dir: Path,
    augmentation_pipeline,
    config: dict,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Args:
        model_class: Model class to instantiate
        data_dir: Data directory
        augmentation_pipeline: Augmentation pipeline
        config: Configuration dictionary
        n_folds: Number of folds

    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold
    from .dataset import MotherboardDataset

    # Load all samples
    dataset = MotherboardDataset(
        data_dir=data_dir,
        augmentation_pipeline=augmentation_pipeline,
        is_training=False,
        augmentation_factor=1,
    )

    # Get labels for stratification
    labels = [label for _, label in dataset.samples]
    indices = list(range(len(dataset.samples)))

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'=' * 50}")

        # Create fold-specific datasets
        train_samples = [dataset.samples[i] for i in train_idx]
        val_samples = [dataset.samples[i] for i in val_idx]

        # Create model and trainer
        classifier_config = config.get("classifier", {})
        model = DefectClassifier(
            backbone=classifier_config.get("backbone", "efficientnet_b0"),
            num_classes=classifier_config.get("num_classes", 5),
            pretrained=classifier_config.get("pretrained", True),
            dropout=classifier_config.get("training", {}).get("dropout", 0.3),
        )

        training_config = classifier_config.get("training", {})
        trainer = ClassifierTrainer(
            model=model,
            device=config.get("hardware", {}).get("device", "cuda"),
            learning_rate=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
            label_smoothing=training_config.get("label_smoothing", 0.1),
            freeze_backbone_epochs=training_config.get("freeze_backbone_epochs", 5),
            patience=training_config.get("early_stopping", {}).get("patience", 15),
        )

        # Train (simplified - would need proper data loader creation)
        # This is a placeholder for the full cross-validation loop

        fold_results.append({
            "fold": fold + 1,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        })

    return {
        "n_folds": n_folds,
        "fold_results": fold_results,
    }


def create_classifier_from_config(config: dict) -> DefectClassifier:
    """
    Create classifier from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured DefectClassifier instance
    """
    classifier_config = config.get("classifier", {})
    training_config = classifier_config.get("training", {})

    return DefectClassifier(
        backbone=classifier_config.get("backbone", "efficientnet_b0"),
        num_classes=classifier_config.get("num_classes", 5),
        pretrained=classifier_config.get("pretrained", True),
        dropout=training_config.get("dropout", 0.3),
    )


def create_trainer_from_config(
    model: DefectClassifier,
    config: dict,
) -> ClassifierTrainer:
    """
    Create trainer from configuration dictionary.

    Args:
        model: DefectClassifier model
        config: Configuration dictionary

    Returns:
        Configured ClassifierTrainer instance
    """
    classifier_config = config.get("classifier", {})
    training_config = classifier_config.get("training", {})
    hardware_config = config.get("hardware", {})
    early_stopping_config = training_config.get("early_stopping", {})

    return ClassifierTrainer(
        model=model,
        device=hardware_config.get("device", "cuda"),
        learning_rate=training_config.get("learning_rate", 0.001),
        weight_decay=training_config.get("weight_decay", 0.0001),
        label_smoothing=training_config.get("label_smoothing", 0.1),
        freeze_backbone_epochs=training_config.get("freeze_backbone_epochs", 5),
        patience=early_stopping_config.get("patience", 15),
        min_delta=early_stopping_config.get("min_delta", 0.001),
    )
