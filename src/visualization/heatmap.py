"""
Heatmap visualization and combination utilities.
Combines PatchCore anomaly maps with Grad-CAM for better localization.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class HeatmapProcessor:
    """
    Processor for combining and visualizing heatmaps from different sources.
    """

    def __init__(
        self,
        patchcore_weight: float = 0.6,
        classifier_weight: float = 0.4,
        colormap: str = "jet",
    ):
        """
        Initialize heatmap processor.

        Args:
            patchcore_weight: Weight for PatchCore anomaly heatmap
            classifier_weight: Weight for Grad-CAM heatmap
            colormap: Matplotlib colormap name
        """
        self.patchcore_weight = patchcore_weight
        self.classifier_weight = classifier_weight
        self.colormap = colormap

    def normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        Normalize heatmap to [0, 1] range.

        Args:
            heatmap: Input heatmap

        Returns:
            Normalized heatmap
        """
        min_val = heatmap.min()
        max_val = heatmap.max()

        if max_val - min_val < 1e-8:
            return np.zeros_like(heatmap)

        return (heatmap - min_val) / (max_val - min_val)

    def combine_heatmaps(
        self,
        patchcore_heatmap: np.ndarray,
        gradcam_heatmap: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Combine PatchCore and Grad-CAM heatmaps.

        Args:
            patchcore_heatmap: Anomaly heatmap from PatchCore (already [0,1])
            gradcam_heatmap: Localization heatmap from Grad-CAM (already [0,1])
            target_size: Target output size (H, W)

        Returns:
            Combined heatmap
        """
        # Determine target size
        if target_size is None:
            target_size = patchcore_heatmap.shape[:2]

        # Resize heatmaps to same size
        pc_resized = cv2.resize(
            patchcore_heatmap,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        gc_resized = cv2.resize(
            gradcam_heatmap,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_LINEAR
        )

        # Use GradCAM as spatial attention mask on PatchCore scores
        # This focuses PatchCore anomalies where classifier sees defects
        combined = pc_resized * (0.5 + 0.5 * gc_resized)

        # Clip to [0, 1]
        combined = np.clip(combined, 0, 1)

        return combined

    def apply_colormap(
        self,
        heatmap: np.ndarray,
        colormap: Optional[str] = None,
    ) -> np.ndarray:
        """
        Apply colormap to heatmap.

        Args:
            heatmap: Input heatmap (H, W) in [0, 1]
            colormap: Colormap name (default: self.colormap)

        Returns:
            Colored heatmap (H, W, 3) in RGB, uint8
        """
        if colormap is None:
            colormap = self.colormap

        # Use OpenCV colormap
        colormap_cv = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "inferno": cv2.COLORMAP_INFERNO,
            "turbo": cv2.COLORMAP_TURBO,
            "viridis": cv2.COLORMAP_VIRIDIS,
        }

        cv_colormap = colormap_cv.get(colormap.lower(), cv2.COLORMAP_JET)

        # Convert to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(heatmap_uint8, cv_colormap)

        # Convert BGR to RGB
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        return colored

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay heatmap on image.

        Args:
            image: Original image (H, W, 3) RGB, uint8 or float32
            heatmap: Heatmap (H, W) in [0, 1]
            alpha: Heatmap transparency

        Returns:
            Overlaid image (H, W, 3) RGB, uint8
        """
        # Ensure image is uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        # Resize heatmap to match image
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(
                heatmap,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        # Apply colormap
        colored_heatmap = self.apply_colormap(heatmap)

        # Blend
        overlay = cv2.addWeighted(
            image, 1 - alpha,
            colored_heatmap, alpha,
            0
        )

        return overlay

    def create_comparison_figure(
        self,
        image: np.ndarray,
        patchcore_heatmap: np.ndarray,
        gradcam_heatmap: np.ndarray,
        combined_heatmap: np.ndarray,
        title: str = "",
        save_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create a figure comparing different heatmaps.

        Args:
            image: Original image
            patchcore_heatmap: PatchCore anomaly heatmap
            gradcam_heatmap: Grad-CAM heatmap
            combined_heatmap: Combined heatmap
            title: Figure title
            save_path: Path to save figure

        Returns:
            Figure as numpy array
        """
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Ensure image is uint8
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        # PatchCore heatmap
        pc_overlay = self.overlay_heatmap(image, patchcore_heatmap)
        axes[1].imshow(pc_overlay)
        axes[1].set_title("PatchCore Anomaly")
        axes[1].axis("off")

        # Grad-CAM heatmap
        gc_overlay = self.overlay_heatmap(image, gradcam_heatmap)
        axes[2].imshow(gc_overlay)
        axes[2].set_title("Grad-CAM")
        axes[2].axis("off")

        # Combined heatmap
        combined_overlay = self.overlay_heatmap(image, combined_heatmap)
        axes[3].imshow(combined_overlay)
        axes[3].set_title("Combined")
        axes[3].axis("off")

        if title:
            fig.suptitle(title, fontsize=14)

        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()
        figure_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return figure_array


class DefectVisualizer:
    """
    Visualizer for defect detection results.
    Creates comprehensive visualizations with bounding boxes and labels.
    """

    def __init__(
        self,
        class_names: List[str],
        colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ):
        """
        Initialize defect visualizer.

        Args:
            class_names: List of defect class names
            colors: Dictionary mapping class names to RGB colors
        """
        self.class_names = class_names

        # Default colors for each class
        if colors is None:
            default_colors = [
                (255, 0, 0),    # Red - burn marks
                (0, 255, 0),    # Green - reuse marks
                (0, 0, 255),    # Blue - liquid damage
                (255, 255, 0),  # Yellow - label tampering
                (255, 0, 255),  # Magenta - other tampering
            ]
            self.colors = {
                name: default_colors[i % len(default_colors)]
                for i, name in enumerate(class_names)
            }
        else:
            self.colors = colors

    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        boxes: List[Dict[str, Any]],
        line_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.

        Args:
            image: Input image (RGB, uint8)
            boxes: List of box dictionaries with 'bbox', 'class', 'confidence'
            line_thickness: Bounding box line thickness

        Returns:
            Image with bounding boxes drawn
        """
        output = image.copy()

        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            class_name = box.get("class", "defect")
            confidence = box.get("confidence", 0.0)

            # Get color
            color = self.colors.get(class_name, (255, 0, 0))

            # Draw rectangle
            cv2.rectangle(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                line_thickness
            )

            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )

            # Label background
            cv2.rectangle(
                output,
                (int(x1), int(y1) - label_size[1] - 10),
                (int(x1) + label_size[0] + 5, int(y1)),
                color,
                -1  # Filled
            )

            # Label text
            cv2.putText(
                output,
                label,
                (int(x1) + 2, int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return output

    def create_detection_visualization(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        boxes: List[Dict[str, Any]],
        anomaly_score: float,
        classification: str,
        confidence: float,
        save_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Create comprehensive detection visualization.

        Args:
            image: Original image
            heatmap: Anomaly heatmap
            boxes: Detected bounding boxes
            anomaly_score: Overall anomaly score
            classification: Predicted class
            confidence: Classification confidence
            save_path: Path to save visualization

        Returns:
            Visualization image
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Original image with boxes
        image_with_boxes = self.draw_bounding_boxes(image, boxes)
        axes[0].imshow(image_with_boxes)
        axes[0].set_title(f"Detection: {classification} ({confidence:.2f})")
        axes[0].axis("off")

        # Heatmap
        heatmap_processor = HeatmapProcessor()
        heatmap_overlay = heatmap_processor.overlay_heatmap(image, heatmap)
        axes[1].imshow(heatmap_overlay)
        axes[1].set_title(f"Anomaly Score: {anomaly_score:.2f}")
        axes[1].axis("off")

        # Heatmap with boxes
        heatmap_with_boxes = self.draw_bounding_boxes(heatmap_overlay, boxes)
        axes[2].imshow(heatmap_with_boxes)
        axes[2].set_title("Combined View")
        axes[2].axis("off")

        plt.tight_layout()

        # Convert to numpy array
        fig.canvas.draw()
        figure_array = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        return figure_array


def save_heatmap(
    heatmap: np.ndarray,
    path: Union[str, Path],
    colormap: str = "jet",
) -> None:
    """
    Save heatmap as image file.

    Args:
        heatmap: Heatmap array (H, W) in [0, 1]
        path: Output file path
        colormap: Colormap name
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    processor = HeatmapProcessor(colormap=colormap)
    colored = processor.apply_colormap(heatmap)

    # Convert RGB to BGR for OpenCV
    cv2.imwrite(str(path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))


def create_heatmap_processor_from_config(config: dict) -> HeatmapProcessor:
    """
    Create HeatmapProcessor from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured HeatmapProcessor instance
    """
    inference_config = config.get("inference", {})
    fusion_config = inference_config.get("fusion", {})

    return HeatmapProcessor(
        patchcore_weight=fusion_config.get("patchcore_weight", 0.6),
        classifier_weight=fusion_config.get("classifier_weight", 0.4),
    )
