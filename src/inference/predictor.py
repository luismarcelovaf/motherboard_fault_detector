"""
Unified inference pipeline combining PatchCore anomaly detection
and EfficientNet classification for motherboard fault detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Union
import yaml
from tqdm import tqdm

from ..models.patchcore import PatchCoreModel
from ..models.classifier import DefectClassifier
from ..visualization.gradcam import GradCAMWrapper
from ..visualization.heatmap import HeatmapProcessor, save_heatmap, DefectVisualizer
from ..data.preprocessing import ImagePreprocessor
from ..data.dataset import CLASS_NAMES
from .postprocessing import (
    HeatmapToBoundingBox,
    DetectionResult,
    BoundingBox,
    ResultFormatter,
)


class FaultDetector:
    """
    Unified fault detection pipeline combining:
    - PatchCore for anomaly detection
    - EfficientNet for defect classification
    - Grad-CAM++ for localization visualization
    """

    def __init__(
        self,
        patchcore_model: PatchCoreModel,
        classifier_model: DefectClassifier,
        device: str = "cuda",
        anomaly_threshold: float = 0.5,
        patchcore_weight: float = 0.6,
        classifier_weight: float = 0.4,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize the fault detector.

        Args:
            patchcore_model: Trained PatchCore model
            classifier_model: Trained defect classifier
            device: Device to run inference on
            anomaly_threshold: Threshold for anomaly decision
            patchcore_weight: Weight for PatchCore heatmap in fusion
            classifier_weight: Weight for Grad-CAM heatmap in fusion
            class_names: List of defect class names
        """
        self.device = device

        # Models
        self.patchcore = patchcore_model
        self.classifier = classifier_model.to(device)
        self.classifier.eval()

        # Grad-CAM
        self.gradcam = GradCAMWrapper(
            model=self.classifier,
            method="gradcam++",
            device=device,
        )

        # Processing components
        self.heatmap_processor = HeatmapProcessor(
            patchcore_weight=patchcore_weight,
            classifier_weight=classifier_weight,
        )
        self.bbox_converter = HeatmapToBoundingBox()
        self.preprocessor = ImagePreprocessor()

        # Parameters
        self.anomaly_threshold = anomaly_threshold
        self.class_names = class_names or CLASS_NAMES

        # Visualization
        self.visualizer = DefectVisualizer(class_names=self.class_names)

    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (256, 256),
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for model input.

        Args:
            image: Input image (BGR or RGB)
            target_size: Target size for preprocessing

        Returns:
            Tuple of (normalized tensor, RGB image for visualization)
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR from cv2
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        # Resize
        image_resized = cv2.resize(
            image_rgb,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize for model
        image_float = image_resized.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_float - mean) / std

        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()

        return tensor, image_resized

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        return_visualization: bool = False,
    ) -> Dict[str, Any]:
        """
        Run full prediction pipeline on a single image.

        Args:
            image: Input image (numpy array or path)
            return_visualization: Whether to include visualization in output

        Returns:
            Dictionary with prediction results
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Could not load image: {image}")

        # Preprocess for PatchCore (256x256)
        patchcore_tensor, image_rgb_256 = self.preprocess_image(
            image, target_size=(256, 256)
        )

        # Preprocess for classifier (224x224)
        classifier_tensor, image_rgb_224 = self.preprocess_image(
            image, target_size=(224, 224)
        )

        # Move to device
        patchcore_tensor = patchcore_tensor.to(self.device)
        classifier_tensor = classifier_tensor.to(self.device)

        # ===== PatchCore Anomaly Detection =====
        anomaly_score, patchcore_heatmap = self.patchcore.predict(patchcore_tensor)

        # ===== Classification =====
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(classifier_tensor.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            predicted_class = probs.argmax(dim=1).item()
            confidence = probs[0, predicted_class].item()

        # ===== Grad-CAM =====
        gradcam_heatmap = self.gradcam.generate_cam(
            classifier_tensor,
            target_class=predicted_class
        )

        # ===== Combine Heatmaps =====
        combined_heatmap = self.heatmap_processor.combine_heatmaps(
            patchcore_heatmap,
            gradcam_heatmap,
            target_size=image_rgb_256.shape[:2],
        )

        # ===== Extract Bounding Boxes =====
        is_anomaly = anomaly_score > self.anomaly_threshold
        defect_type = self.class_names[predicted_class] if is_anomaly else "normal"

        bounding_boxes = []
        if is_anomaly:
            bounding_boxes = self.bbox_converter.convert(
                combined_heatmap,
                defect_type=defect_type,
            )

        # ===== Format Result =====
        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "classification": defect_type,
            "confidence": float(confidence),
            "class_probabilities": {
                self.class_names[i]: float(probs[0, i])
                for i in range(len(self.class_names))
            },
            "defects": [box.to_dict() for box in bounding_boxes],
            "heatmaps": {
                "patchcore": patchcore_heatmap,
                "gradcam": gradcam_heatmap,
                "combined": combined_heatmap,
            },
        }

        # ===== Visualization =====
        if return_visualization:
            # Overlay heatmap on image
            heatmap_overlay = self.heatmap_processor.overlay_heatmap(
                image_rgb_256 / 255.0 if image_rgb_256.max() > 1 else image_rgb_256,
                combined_heatmap,
            )

            # Draw bounding boxes
            if bounding_boxes:
                image_with_boxes = self.visualizer.draw_bounding_boxes(
                    image_rgb_256,
                    [box.to_dict() for box in bounding_boxes],
                )
            else:
                image_with_boxes = image_rgb_256

            result["visualization"] = {
                "heatmap_overlay": heatmap_overlay,
                "image_with_boxes": image_with_boxes,
                "original": image_rgb_256,
            }

        return result

    def predict_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        save_results: bool = False,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run prediction on a batch of images.

        Args:
            images: List of images or paths
            save_results: Whether to save results to disk
            output_dir: Directory for saving results

        Returns:
            List of prediction results
        """
        results = []

        for i, image in enumerate(tqdm(images, desc="Processing images")):
            try:
                result = self.predict(image, return_visualization=save_results)

                if save_results and output_dir:
                    self._save_result(result, image, output_dir, i)

                # Remove heavy data if not saving
                if not save_results:
                    result.pop("heatmaps", None)

                results.append(result)

            except Exception as e:
                print(f"Error processing image {i}: {e}")
                results.append({"error": str(e)})

        return results

    def _save_result(
        self,
        result: Dict[str, Any],
        image_source: Union[np.ndarray, str, Path],
        output_dir: Path,
        index: int,
    ) -> None:
        """Save prediction result to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get base name
        if isinstance(image_source, (str, Path)):
            base_name = Path(image_source).stem
        else:
            base_name = f"image_{index:04d}"

        # Save heatmap
        if "heatmaps" in result:
            heatmap_path = output_dir / f"{base_name}_heatmap.png"
            save_heatmap(result["heatmaps"]["combined"], heatmap_path)
            result["heatmap_path"] = str(heatmap_path)

        # Save visualization
        if "visualization" in result:
            vis_path = output_dir / f"{base_name}_visualization.png"
            cv2.imwrite(
                str(vis_path),
                cv2.cvtColor(result["visualization"]["heatmap_overlay"], cv2.COLOR_RGB2BGR)
            )

        # Save JSON result
        import json
        result_copy = {k: v for k, v in result.items()
                       if k not in ["heatmaps", "visualization"]}
        json_path = output_dir / f"{base_name}_result.json"
        with open(json_path, "w") as f:
            json.dump(result_copy, f, indent=2)

    def evaluate(
        self,
        test_images: List[Union[np.ndarray, str, Path]],
        ground_truth_labels: List[bool],  # True = anomaly
        ground_truth_classes: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test set.

        Args:
            test_images: List of test images
            ground_truth_labels: Ground truth anomaly labels
            ground_truth_classes: Ground truth class labels (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import (
            roc_auc_score, f1_score, accuracy_score,
            precision_score, recall_score
        )

        predictions = self.predict_batch(test_images, save_results=False)

        # Extract predictions
        pred_scores = [p.get("anomaly_score", 0) for p in predictions if "error" not in p]
        pred_labels = [p.get("is_anomaly", False) for p in predictions if "error" not in p]

        # Filter ground truth for successful predictions
        valid_gt = [gt for i, gt in enumerate(ground_truth_labels)
                    if "error" not in predictions[i]]

        # Calculate metrics
        metrics = {}

        if len(pred_scores) > 0:
            # Anomaly detection metrics
            metrics["auroc"] = roc_auc_score(valid_gt, pred_scores)
            metrics["accuracy"] = accuracy_score(valid_gt, pred_labels)
            metrics["precision"] = precision_score(valid_gt, pred_labels, zero_division=0)
            metrics["recall"] = recall_score(valid_gt, pred_labels, zero_division=0)
            metrics["f1"] = f1_score(valid_gt, pred_labels, zero_division=0)

        # Classification metrics (if ground truth classes provided)
        if ground_truth_classes is not None:
            pred_classes = [
                self.class_names.index(p.get("classification", "normal"))
                if p.get("classification", "normal") in self.class_names else -1
                for p in predictions if "error" not in p
            ]
            valid_gt_classes = [gt for i, gt in enumerate(ground_truth_classes)
                                if "error" not in predictions[i]]

            # Only evaluate on anomalies
            anomaly_pred = [pc for pc, pl in zip(pred_classes, pred_labels) if pl]
            anomaly_gt = [gc for gc, gl in zip(valid_gt_classes, valid_gt) if gl]

            if anomaly_pred and anomaly_gt:
                metrics["classification_accuracy"] = accuracy_score(anomaly_gt, anomaly_pred)
                metrics["classification_f1"] = f1_score(
                    anomaly_gt, anomaly_pred, average="macro", zero_division=0
                )

        return metrics


def load_detector_from_config(
    config_path: Union[str, Path],
    patchcore_checkpoint: Union[str, Path],
    classifier_checkpoint: Union[str, Path],
) -> FaultDetector:
    """
    Load FaultDetector from configuration and checkpoints.

    Args:
        config_path: Path to config.yaml
        patchcore_checkpoint: Path to PatchCore model checkpoint
        classifier_checkpoint: Path to classifier checkpoint

    Returns:
        Configured FaultDetector instance
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Auto-detect device (fallback to CPU if CUDA not available)
    configured_device = config.get("hardware", {}).get("device", "cuda")
    if configured_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = "cpu"
    else:
        device = configured_device

    # Load PatchCore
    patchcore_config = config.get("patchcore", {})
    patchcore = PatchCoreModel(
        backbone=patchcore_config.get("backbone", "wide_resnet50_2"),
        layers_to_extract=patchcore_config.get("layers_to_extract", ["layer2", "layer3"]),
        num_neighbors=patchcore_config.get("num_neighbors", 9),
        input_size=tuple(patchcore_config.get("input_size", [256, 256])),
        device=device,
    )
    patchcore.load(patchcore_checkpoint)

    # Load classifier
    classifier_config = config.get("classifier", {})
    classifier = DefectClassifier(
        backbone=classifier_config.get("backbone", "efficientnet_b0"),
        num_classes=classifier_config.get("num_classes", 5),
        pretrained=False,  # Will load from checkpoint
    )

    checkpoint = torch.load(classifier_checkpoint, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])

    # Create detector
    inference_config = config.get("inference", {})
    fusion_config = inference_config.get("fusion", {})
    threshold_config = patchcore_config.get("thresholds", {})

    detector = FaultDetector(
        patchcore_model=patchcore,
        classifier_model=classifier,
        device=device,
        anomaly_threshold=threshold_config.get("anomaly_score", 0.5),
        patchcore_weight=fusion_config.get("patchcore_weight", 0.6),
        classifier_weight=fusion_config.get("classifier_weight", 0.4),
        class_names=config.get("data", {}).get("defect_classes", CLASS_NAMES),
    )

    return detector


def create_detector(
    patchcore_model: PatchCoreModel,
    classifier_model: DefectClassifier,
    config: dict,
) -> FaultDetector:
    """
    Create FaultDetector from models and configuration.

    Args:
        patchcore_model: Trained PatchCore model
        classifier_model: Trained classifier model
        config: Configuration dictionary

    Returns:
        Configured FaultDetector instance
    """
    hardware_config = config.get("hardware", {})
    inference_config = config.get("inference", {})
    fusion_config = inference_config.get("fusion", {})
    patchcore_config = config.get("patchcore", {})
    threshold_config = patchcore_config.get("thresholds", {})

    return FaultDetector(
        patchcore_model=patchcore_model,
        classifier_model=classifier_model,
        device=hardware_config.get("device", "cuda"),
        anomaly_threshold=threshold_config.get("anomaly_score", 0.5),
        patchcore_weight=fusion_config.get("patchcore_weight", 0.6),
        classifier_weight=fusion_config.get("classifier_weight", 0.4),
        class_names=config.get("data", {}).get("defect_classes", CLASS_NAMES),
    )
