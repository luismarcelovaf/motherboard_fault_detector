"""
Post-processing utilities for defect detection.
Converts heatmaps to bounding boxes and handles result formatting.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import json


@dataclass
class BoundingBox:
    """Represents a detected defect bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    defect_type: str

    @property
    def area(self) -> int:
        """Calculate bounding box area."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
            "class": self.defect_type,
        }


@dataclass
class DetectionResult:
    """Complete detection result for a single image."""
    is_anomaly: bool
    anomaly_score: float
    classification: str
    confidence: float
    defects: List[BoundingBox]
    heatmap_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": float(self.anomaly_score),
            "classification": self.classification,
            "confidence": float(self.confidence),
            "defects": [d.to_dict() for d in self.defects],
            "heatmap_path": self.heatmap_path,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class HeatmapToBoundingBox:
    """
    Converts anomaly heatmaps to bounding boxes.
    Uses thresholding, morphological operations, and contour detection.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_area: int = 100,
        max_boxes: int = 10,
        morphology_kernel_size: int = 5,
        nms_threshold: float = 0.3,
    ):
        """
        Initialize converter.

        Args:
            threshold: Heatmap threshold for binarization
            min_area: Minimum contour area to be considered
            max_boxes: Maximum number of bounding boxes to return
            morphology_kernel_size: Kernel size for morphological operations
            nms_threshold: IoU threshold for non-maximum suppression
        """
        self.threshold = threshold
        self.min_area = min_area
        self.max_boxes = max_boxes
        self.morphology_kernel_size = morphology_kernel_size
        self.nms_threshold = nms_threshold

    def _binarize(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Binarize heatmap using threshold.

        Args:
            heatmap: Normalized heatmap [0, 1]

        Returns:
            Binary mask
        """
        return (heatmap > self.threshold).astype(np.uint8) * 255

    def _apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up binary mask.

        Args:
            binary_mask: Binary mask

        Returns:
            Cleaned binary mask
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )

        # Close small gaps
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        return opened

    def _find_contours(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """
        Find contours in binary mask.

        Args:
            binary_mask: Binary mask

        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _contour_to_bbox(
        self,
        contour: np.ndarray,
        heatmap: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Convert contour to bounding box with confidence.

        Args:
            contour: Contour points
            heatmap: Original heatmap for confidence calculation

        Returns:
            Tuple of (x1, y1, x2, y2, confidence) or None if invalid
        """
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Check minimum area
        area = w * h
        if area < self.min_area:
            return None

        # Calculate confidence as mean heatmap value in bounding box
        roi = heatmap[y:y+h, x:x+w]
        confidence = float(np.mean(roi))

        return (x, y, x + w, y + h, confidence)

    def _nms(
        self,
        boxes: List[Tuple[int, int, int, int, float]],
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Apply non-maximum suppression.

        Args:
            boxes: List of (x1, y1, x2, y2, confidence) tuples

        Returns:
            Filtered boxes after NMS
        """
        if len(boxes) == 0:
            return []

        # Sort by confidence (descending)
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

        keep = []

        while boxes:
            # Keep the box with highest confidence
            current = boxes.pop(0)
            keep.append(current)

            # Filter remaining boxes
            remaining = []
            for box in boxes:
                iou = self._calculate_iou(current[:4], box[:4])
                if iou < self.nms_threshold:
                    remaining.append(box)
            boxes = remaining

        return keep[:self.max_boxes]

    def _calculate_iou(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int],
    ) -> float:
        """
        Calculate Intersection over Union between two boxes.

        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)

        Returns:
            IoU value
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def convert(
        self,
        heatmap: np.ndarray,
        defect_type: str = "defect",
    ) -> List[BoundingBox]:
        """
        Convert heatmap to bounding boxes.

        Args:
            heatmap: Anomaly heatmap (H, W) normalized to [0, 1]
            defect_type: Type of defect for labeling

        Returns:
            List of BoundingBox objects
        """
        # Binarize
        binary = self._binarize(heatmap)

        # Apply morphology
        cleaned = self._apply_morphology(binary)

        # Find contours
        contours = self._find_contours(cleaned)

        # Convert to bounding boxes
        boxes = []
        for contour in contours:
            result = self._contour_to_bbox(contour, heatmap)
            if result is not None:
                boxes.append(result)

        # Apply NMS
        filtered_boxes = self._nms(boxes)

        # Create BoundingBox objects
        return [
            BoundingBox(
                x1=int(box[0]),
                y1=int(box[1]),
                x2=int(box[2]),
                y2=int(box[3]),
                confidence=box[4],
                defect_type=defect_type,
            )
            for box in filtered_boxes
        ]


class AdaptiveThreshold:
    """
    Adaptive threshold selection based on anomaly score distribution.
    """

    def __init__(
        self,
        method: str = "otsu",
        percentile: float = 95,
    ):
        """
        Initialize adaptive threshold.

        Args:
            method: Threshold method ('otsu', 'percentile', 'mean')
            percentile: Percentile for percentile method
        """
        self.method = method
        self.percentile = percentile

    def compute_threshold(self, heatmap: np.ndarray) -> float:
        """
        Compute adaptive threshold for heatmap.

        Args:
            heatmap: Anomaly heatmap

        Returns:
            Computed threshold value
        """
        if self.method == "otsu":
            # Otsu's method
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            threshold, _ = cv2.threshold(
                heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return threshold / 255.0

        elif self.method == "percentile":
            # Percentile method
            return np.percentile(heatmap, self.percentile)

        elif self.method == "mean":
            # Mean + std method
            return np.mean(heatmap) + np.std(heatmap)

        else:
            raise ValueError(f"Unknown threshold method: {self.method}")


class ResultFormatter:
    """
    Formats detection results for output.
    """

    def __init__(
        self,
        class_names: List[str],
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize formatter.

        Args:
            class_names: List of defect class names
            output_dir: Directory for output files
        """
        self.class_names = class_names
        self.output_dir = Path(output_dir) if output_dir else None

    def format_result(
        self,
        anomaly_score: float,
        anomaly_threshold: float,
        class_predictions: np.ndarray,
        class_confidences: np.ndarray,
        bounding_boxes: List[BoundingBox],
        heatmap_path: Optional[str] = None,
    ) -> DetectionResult:
        """
        Format raw predictions into DetectionResult.

        Args:
            anomaly_score: Overall anomaly score
            anomaly_threshold: Threshold for anomaly decision
            class_predictions: Predicted class indices
            class_confidences: Class prediction confidences
            bounding_boxes: List of detected bounding boxes
            heatmap_path: Path to saved heatmap

        Returns:
            Formatted DetectionResult
        """
        # Determine if anomaly
        is_anomaly = anomaly_score > anomaly_threshold

        # Get top classification
        if len(class_predictions) > 0:
            top_class_idx = int(class_predictions[0])
            classification = self.class_names[top_class_idx]
            confidence = float(class_confidences[0])
        else:
            classification = "normal"
            confidence = 1.0 - anomaly_score

        return DetectionResult(
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            classification=classification,
            confidence=confidence,
            defects=bounding_boxes,
            heatmap_path=heatmap_path,
        )

    def save_result(
        self,
        result: DetectionResult,
        image_name: str,
    ) -> Path:
        """
        Save detection result to JSON file.

        Args:
            result: Detection result
            image_name: Name of input image

        Returns:
            Path to saved file
        """
        if self.output_dir is None:
            raise ValueError("Output directory not set")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        output_path = self.output_dir / f"{Path(image_name).stem}_result.json"

        with open(output_path, "w") as f:
            f.write(result.to_json())

        return output_path

    def create_summary(
        self,
        results: List[DetectionResult],
    ) -> Dict[str, Any]:
        """
        Create summary statistics from multiple results.

        Args:
            results: List of detection results

        Returns:
            Summary dictionary
        """
        total = len(results)
        num_anomalies = sum(1 for r in results if r.is_anomaly)

        # Class distribution
        class_counts = {name: 0 for name in self.class_names}
        for r in results:
            if r.is_anomaly and r.classification in class_counts:
                class_counts[r.classification] += 1

        return {
            "total_images": total,
            "anomalies_detected": num_anomalies,
            "normal_count": total - num_anomalies,
            "anomaly_rate": num_anomalies / total if total > 0 else 0,
            "class_distribution": class_counts,
            "average_anomaly_score": np.mean([r.anomaly_score for r in results]),
            "average_confidence": np.mean([r.confidence for r in results if r.is_anomaly] or [0]),
        }


def create_postprocessor_from_config(config: dict) -> HeatmapToBoundingBox:
    """
    Create HeatmapToBoundingBox from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured HeatmapToBoundingBox instance
    """
    inference_config = config.get("inference", {})
    heatmap_config = inference_config.get("heatmap", {})
    bbox_config = inference_config.get("bbox", {})

    return HeatmapToBoundingBox(
        threshold=heatmap_config.get("threshold", 0.5),
        min_area=bbox_config.get("min_area", 100),
        max_boxes=bbox_config.get("max_boxes", 10),
        morphology_kernel_size=heatmap_config.get("morphology_kernel", 5),
        nms_threshold=bbox_config.get("nms_threshold", 0.3),
    )
