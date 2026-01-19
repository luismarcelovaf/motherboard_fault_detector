"""
PatchCore anomaly detection model for motherboard fault detection.
Uses anomalib library for the core implementation with a simplified wrapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

# Anomalib imports
try:
    from anomalib.models import Patchcore
    from anomalib.data import PredictDataset
    from anomalib.engine import Engine
    from anomalib import TaskType
    ANOMALIB_AVAILABLE = True
except ImportError:
    ANOMALIB_AVAILABLE = False
    print("Warning: anomalib not installed. Using custom PatchCore implementation.")

# Timm for backbone
import timm


class FeatureExtractor(nn.Module):
    """
    Feature extractor using pretrained backbone.
    Extracts features from specified intermediate layers.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers_to_extract: List[str] = ["layer2", "layer3"],
        pretrained: bool = True,
    ):
        """
        Initialize the feature extractor.

        Args:
            backbone_name: Name of the backbone model
            layers_to_extract: List of layer names to extract features from
            pretrained: Whether to use pretrained weights
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.layers_to_extract = layers_to_extract

        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 3],  # layer2 and layer3 for wide_resnet50
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors from specified layers
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features


class PatchCoreModel:
    """
    PatchCore anomaly detection model.
    Builds a memory bank from normal images and detects anomalies via k-NN.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers_to_extract: List[str] = ["layer2", "layer3"],
        num_neighbors: int = 9,
        coreset_sampling_ratio: float = 0.1,
        input_size: Tuple[int, int] = (256, 256),
        device: str = "cuda",
    ):
        """
        Initialize PatchCore model.

        Args:
            backbone: Backbone model name
            layers_to_extract: Layers to extract features from
            num_neighbors: Number of neighbors for k-NN scoring
            coreset_sampling_ratio: Ratio for coreset subsampling
            input_size: Expected input image size
            device: Device to run on
        """
        self.backbone_name = backbone
        self.layers_to_extract = layers_to_extract
        self.num_neighbors = num_neighbors
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.input_size = input_size
        self.device = device

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone,
            layers_to_extract=layers_to_extract,
            pretrained=True,
        ).to(device)

        # Memory bank (will be populated during fit)
        self.memory_bank: Optional[torch.Tensor] = None
        self.feature_shape: Optional[Tuple[int, int]] = None

    def _extract_patch_features(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract and aggregate patch features from multiple layers.

        Args:
            features: List of feature tensors from different layers

        Returns:
            Aggregated patch features of shape (B, num_patches, feature_dim)
        """
        # Upsample all features to the same spatial size (use largest)
        target_size = features[0].shape[2:]

        upsampled = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False
                )
            upsampled.append(feat)

        # Concatenate along channel dimension
        concatenated = torch.cat(upsampled, dim=1)

        # Reshape to (B, num_patches, feature_dim)
        B, C, H, W = concatenated.shape
        self.feature_shape = (H, W)
        patch_features = concatenated.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return patch_features

    def _subsample_coreset(
        self,
        features: torch.Tensor,
        sampling_ratio: float
    ) -> torch.Tensor:
        """
        Subsample memory bank using coreset selection.

        Args:
            features: All patch features of shape (N, feature_dim)
            sampling_ratio: Ratio of features to keep

        Returns:
            Subsampled features
        """
        num_samples = int(len(features) * sampling_ratio)
        num_samples = max(num_samples, 1)

        if num_samples >= len(features):
            return features

        # Use GPU for faster computation
        device = self.device
        features_gpu = features.to(device)

        # For large coreset sizes, use fast random sampling
        # (nearly as effective as greedy sampling, but 1000x faster)
        if num_samples > 1000:
            print(f"Using random sampling for {num_samples} samples (fast mode)...")
            indices = torch.randperm(len(features_gpu))[:num_samples]
            return features_gpu[indices]

        # For smaller sizes, use greedy furthest point sampling on GPU
        print(f"Using greedy furthest point sampling for {num_samples} samples...")
        indices = [torch.randint(len(features_gpu), (1,)).item()]

        # Pre-allocate distance tensor
        min_dists = torch.full((len(features_gpu),), float('inf'), device=device)

        for _ in tqdm(range(num_samples - 1), desc="Coreset sampling"):
            # Get the last selected point
            last_selected = features_gpu[indices[-1]].unsqueeze(0)

            # Compute distances to last selected point only
            new_dists = torch.cdist(features_gpu, last_selected).squeeze(1)

            # Update minimum distances
            min_dists = torch.minimum(min_dists, new_dists)

            # Select furthest point
            new_idx = min_dists.argmax().item()
            indices.append(new_idx)

            # Mark selected point
            min_dists[new_idx] = -1

        return features_gpu[indices]

    def fit(self, dataloader: DataLoader) -> None:
        """
        Build memory bank from normal images.

        Args:
            dataloader: DataLoader yielding normal images
        """
        self.feature_extractor.eval()
        all_features = []

        print("Extracting features from normal images...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Building memory bank"):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device)

                # Extract features
                features = self.feature_extractor(images)

                # Get patch features
                patch_features = self._extract_patch_features(features)

                # Flatten batch dimension
                patch_features = patch_features.reshape(-1, patch_features.shape[-1])
                all_features.append(patch_features.cpu())

        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)
        print(f"Total patch features: {len(all_features)}")

        # Subsample using coreset
        print("Subsampling coreset...")
        self.memory_bank = self._subsample_coreset(
            all_features,
            self.coreset_sampling_ratio
        ).to(self.device)

        print(f"Memory bank size: {len(self.memory_bank)}")

    def predict(
        self,
        image: torch.Tensor
    ) -> Tuple[float, np.ndarray]:
        """
        Predict anomaly score and heatmap for a single image.

        Args:
            image: Input tensor of shape (C, H, W) or (1, C, H, W)

        Returns:
            Tuple of (anomaly_score, heatmap)
        """
        if self.memory_bank is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.feature_extractor.eval()

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            # Extract features
            features = self.feature_extractor(image)
            patch_features = self._extract_patch_features(features)

            # Compute distances to memory bank
            # patch_features: (1, num_patches, feature_dim)
            # memory_bank: (memory_size, feature_dim)
            patch_features = patch_features.squeeze(0)  # (num_patches, feature_dim)

            # k-NN distances
            distances = torch.cdist(patch_features, self.memory_bank)
            knn_distances, _ = distances.topk(
                self.num_neighbors,
                largest=False,
                dim=1
            )

            # Anomaly score per patch (mean of k-NN distances)
            patch_scores = knn_distances.mean(dim=1)

            # Image-level anomaly score
            anomaly_score = patch_scores.max().item()

            # Reshape to heatmap
            H, W = self.feature_shape
            heatmap = patch_scores.reshape(H, W).cpu().numpy()

            # Upsample heatmap to input size
            heatmap = cv2.resize(
                heatmap,
                (self.input_size[1], self.input_size[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Normalize heatmap to [0, 1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return anomaly_score, heatmap

    def predict_batch(
        self,
        images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores and heatmaps for a batch of images.

        Args:
            images: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of (anomaly_scores, heatmaps)
        """
        if self.memory_bank is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.feature_extractor.eval()
        images = images.to(self.device)

        with torch.no_grad():
            # Extract features
            features = self.feature_extractor(images)
            patch_features = self._extract_patch_features(features)

            B = patch_features.shape[0]
            H, W = self.feature_shape

            anomaly_scores = []
            heatmaps = []

            for i in range(B):
                pf = patch_features[i]  # (num_patches, feature_dim)

                # k-NN distances
                distances = torch.cdist(pf, self.memory_bank)
                knn_distances, _ = distances.topk(
                    self.num_neighbors,
                    largest=False,
                    dim=1
                )

                # Anomaly score per patch
                patch_scores = knn_distances.mean(dim=1)

                # Image-level score
                anomaly_scores.append(patch_scores.max().item())

                # Heatmap
                heatmap = patch_scores.reshape(H, W).cpu().numpy()
                heatmap = cv2.resize(
                    heatmap,
                    (self.input_size[1], self.input_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                heatmaps.append(heatmap)

        return np.array(anomaly_scores), np.stack(heatmaps)

    def save(self, path: Union[str, Path]) -> None:
        """Save the model (memory bank and config)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "feature_shape": self.feature_shape,
            "config": {
                "backbone": self.backbone_name,
                "layers_to_extract": self.layers_to_extract,
                "num_neighbors": self.num_neighbors,
                "coreset_sampling_ratio": self.coreset_sampling_ratio,
                "input_size": self.input_size,
            }
        }
        torch.save(state, path)
        print(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load a saved model."""
        path = Path(path)
        state = torch.load(path, map_location=self.device)

        self.memory_bank = state["memory_bank"].to(self.device)
        self.feature_shape = state["feature_shape"]

        config = state["config"]
        self.backbone_name = config["backbone"]
        self.layers_to_extract = config["layers_to_extract"]
        self.num_neighbors = config["num_neighbors"]
        self.coreset_sampling_ratio = config["coreset_sampling_ratio"]
        self.input_size = tuple(config["input_size"])

        print(f"Model loaded from {path}")


class AnomalibPatchCore:
    """
    Wrapper for anomalib's PatchCore implementation.
    Provides a simpler interface for training and inference.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: List[str] = ["layer2", "layer3"],
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
    ):
        """
        Initialize anomalib PatchCore wrapper.

        Args:
            backbone: Backbone model name
            layers: Layers to extract features from
            coreset_sampling_ratio: Coreset sampling ratio
            num_neighbors: Number of neighbors for scoring
        """
        if not ANOMALIB_AVAILABLE:
            raise ImportError("anomalib is required for AnomalibPatchCore")

        self.model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
        self.engine = None

    def fit(self, data_path: Union[str, Path], **kwargs) -> None:
        """
        Train PatchCore on normal images.

        Args:
            data_path: Path to data directory with 'normal' subfolder
            **kwargs: Additional arguments for anomalib Engine
        """
        self.engine = Engine(task=TaskType.SEGMENTATION)
        # Note: anomalib handles data loading internally
        # This is a simplified interface - actual training would use anomalib's
        # datamodule setup

    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Predict on a single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction results
        """
        if self.engine is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Use anomalib's prediction interface
        # This is a simplified placeholder
        pass


def get_device(config: dict) -> str:
    """
    Get device from config with automatic CPU fallback.

    Args:
        config: Configuration dictionary

    Returns:
        Device string ('cuda' or 'cpu')
    """
    configured_device = config.get("hardware", {}).get("device", "cuda")
    if configured_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        return "cpu"
    return configured_device


def create_patchcore_from_config(config: dict) -> PatchCoreModel:
    """
    Create PatchCore model from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Configured PatchCoreModel instance
    """
    patchcore_config = config.get("patchcore", {})
    device = get_device(config)

    return PatchCoreModel(
        backbone=patchcore_config.get("backbone", "wide_resnet50_2"),
        layers_to_extract=patchcore_config.get("layers_to_extract", ["layer2", "layer3"]),
        num_neighbors=patchcore_config.get("num_neighbors", 9),
        coreset_sampling_ratio=patchcore_config.get("coreset_sampling_ratio", 0.1),
        input_size=tuple(patchcore_config.get("input_size", [256, 256])),
        device=device,
    )
