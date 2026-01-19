"""
Grad-CAM and Grad-CAM++ visualization for defect localization.
Uses pytorch-grad-cam library for robust implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any, Union, Callable
from pathlib import Path

# pytorch-grad-cam imports
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed. Using custom implementation.")


class GradCAMWrapper:
    """
    Wrapper for Grad-CAM and Grad-CAM++ visualization.
    Provides localization heatmaps for classifier predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[nn.Module]] = None,
        method: str = "gradcam++",
        device: str = "cuda",
    ):
        """
        Initialize Grad-CAM wrapper.

        Args:
            model: Classifier model
            target_layers: List of layers to compute CAM for
            method: CAM method ('gradcam', 'gradcam++', 'scorecam')
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.method = method

        # Auto-detect target layers if not provided
        if target_layers is None:
            target_layers = self._find_target_layers()

        self.target_layers = target_layers

        # Initialize CAM object
        if GRADCAM_AVAILABLE:
            self._init_pytorch_gradcam()
        else:
            self._init_custom_gradcam()

    def _find_target_layers(self) -> List[nn.Module]:
        """
        Auto-detect suitable target layers for CAM.
        Typically the last convolutional layer.
        """
        target_layers = []

        # For EfficientNet-style models
        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone

            # Try common layer names
            if hasattr(backbone, "conv_head"):
                target_layers = [backbone.conv_head]
            elif hasattr(backbone, "features"):
                # Get last feature block
                features = backbone.features
                if isinstance(features, nn.Sequential):
                    for layer in reversed(list(features.children())):
                        if isinstance(layer, (nn.Conv2d, nn.Sequential)):
                            target_layers = [layer]
                            break

        if not target_layers:
            # Fallback: find last conv layer
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layers = [module]

        return target_layers

    def _init_pytorch_gradcam(self):
        """Initialize pytorch-grad-cam CAM object."""
        cam_class_map = {
            "gradcam": GradCAM,
            "gradcam++": GradCAMPlusPlus,
            "scorecam": ScoreCAM,
        }

        cam_class = cam_class_map.get(self.method.lower(), GradCAMPlusPlus)
        self.cam = cam_class(
            model=self.model,
            target_layers=self.target_layers,
        )

    def _init_custom_gradcam(self):
        """Initialize custom Grad-CAM implementation."""
        self.cam = None
        self.gradients = None
        self.activations = None

        # Register hooks
        if self.target_layers:
            target_layer = self.target_layers[0]

            def save_gradient(grad):
                self.gradients = grad

            def save_activation(module, input, output):
                self.activations = output
                if output.requires_grad:
                    output.register_hook(save_gradient)

            target_layer.register_forward_hook(save_activation)

    def generate_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate CAM heatmap for an image.

        Args:
            image: Input tensor of shape (C, H, W) or (1, C, H, W)
            target_class: Target class for CAM (None = predicted class)

        Returns:
            CAM heatmap of shape (H, W) normalized to [0, 1]
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        if GRADCAM_AVAILABLE:
            return self._generate_cam_pytorch(image, target_class)
        else:
            return self._generate_cam_custom(image, target_class)

    def _generate_cam_pytorch(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate CAM using pytorch-grad-cam."""
        # Set up targets
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            targets = None

        # Generate CAM
        cam_output = self.cam(input_tensor=image, targets=targets)

        return cam_output[0]

    def _generate_cam_custom(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate CAM using custom implementation (Grad-CAM++)."""
        self.model.eval()

        # Forward pass
        output = self.model(image)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients.data
        activations = self.activations.data

        # Grad-CAM++ weights
        if self.method.lower() == "gradcam++":
            # Second and third order gradients
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)

            # Calculate alpha
            sum_activations = activations.sum(dim=(2, 3), keepdim=True)
            alpha_num = grad_2
            alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
            alpha = alpha_num / alpha_denom

            # Weights
            weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        else:
            # Standard Grad-CAM: global average pooling of gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input size
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        return cam[0, 0].cpu().numpy()

    def visualize(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on image.

        Args:
            image: Original image (RGB, float32, [0,1] or uint8 [0,255])
            cam: CAM heatmap (H, W) normalized to [0, 1]
            alpha: Overlay transparency

        Returns:
            Visualization image (RGB, uint8)
        """
        # Ensure image is float32 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Resize CAM to match image
        if cam.shape[:2] != image.shape[:2]:
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

        if GRADCAM_AVAILABLE:
            visualization = show_cam_on_image(image, cam, use_rgb=True)
        else:
            # Custom visualization
            heatmap = cv2.applyColorMap(
                (cam * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = heatmap.astype(np.float32) / 255.0

            visualization = (1 - alpha) * image + alpha * heatmap
            visualization = (visualization * 255).astype(np.uint8)

        return visualization

    def generate_and_visualize(
        self,
        image_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CAM and visualization in one step.

        Args:
            image_tensor: Normalized input tensor
            original_image: Original unnormalized image (RGB, [0, 255] or [0, 1])
            target_class: Target class for CAM

        Returns:
            Tuple of (cam_heatmap, visualization)
        """
        # Generate CAM
        cam = self.generate_cam(image_tensor, target_class)

        # Create visualization
        visualization = self.visualize(original_image, cam)

        return cam, visualization


class MultiScaleGradCAM:
    """
    Multi-scale Grad-CAM for better localization.
    Combines CAMs from multiple layers.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = "cuda",
    ):
        """
        Initialize multi-scale Grad-CAM.

        Args:
            model: Classifier model
            layer_names: Names of layers to extract CAM from
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        self.layer_names = layer_names

        # Find layers by name
        self.target_layers = []
        for name, module in model.named_modules():
            if name in layer_names:
                self.target_layers.append(module)

        # Create individual CAM objects
        self.cam_objects = []
        if GRADCAM_AVAILABLE:
            for layer in self.target_layers:
                cam = GradCAMPlusPlus(
                    model=self.model,
                    target_layers=[layer],
                )
                self.cam_objects.append(cam)

    def generate_multiscale_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Generate multi-scale CAM by combining CAMs from multiple layers.

        Args:
            image: Input tensor
            target_class: Target class
            weights: Weights for combining layer CAMs (default: equal)

        Returns:
            Combined CAM heatmap
        """
        if not GRADCAM_AVAILABLE:
            # Fallback to single-scale
            wrapper = GradCAMWrapper(self.model, device=self.device)
            return wrapper.generate_cam(image, target_class)

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Generate CAM from each layer
        cams = []
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

        for cam_obj in self.cam_objects:
            cam = cam_obj(input_tensor=image, targets=targets)[0]
            # Resize to common size
            cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
            cams.append(cam)

        # Combine CAMs
        if weights is None:
            weights = [1.0 / len(cams)] * len(cams)

        combined_cam = np.zeros_like(cams[0])
        for cam, weight in zip(cams, weights):
            combined_cam += weight * cam

        # Normalize
        combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min() + 1e-8)

        return combined_cam


def get_gradcam_target_layers(model: nn.Module, model_type: str = "efficientnet") -> List[nn.Module]:
    """
    Get appropriate target layers for Grad-CAM based on model architecture.

    Args:
        model: The classifier model
        model_type: Type of model architecture

    Returns:
        List of target layer modules
    """
    if model_type == "efficientnet":
        # For EfficientNet, use the last convolutional block
        if hasattr(model, "backbone"):
            backbone = model.backbone
            if hasattr(backbone, "conv_head"):
                return [backbone.conv_head]
            elif hasattr(backbone, "blocks"):
                # Last block
                return [list(backbone.blocks.children())[-1]]

    elif model_type == "resnet":
        # For ResNet, use layer4
        if hasattr(model, "backbone"):
            backbone = model.backbone
            if hasattr(backbone, "layer4"):
                return [backbone.layer4]

    # Fallback: find last conv layer
    target = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            target = module

    return [target] if target else []


def create_gradcam_from_config(
    model: nn.Module,
    config: dict,
) -> GradCAMWrapper:
    """
    Create Grad-CAM wrapper from configuration.

    Args:
        model: Classifier model
        config: Configuration dictionary

    Returns:
        Configured GradCAMWrapper instance
    """
    gradcam_config = config.get("gradcam", {})
    hardware_config = config.get("hardware", {})

    # Get target layers
    target_layer_names = gradcam_config.get("target_layers", [])
    target_layers = []

    if target_layer_names:
        for name, module in model.named_modules():
            if name in target_layer_names:
                target_layers.append(module)

    if not target_layers:
        target_layers = None  # Will auto-detect

    return GradCAMWrapper(
        model=model,
        target_layers=target_layers,
        method=gradcam_config.get("method", "gradcam++"),
        device=hardware_config.get("device", "cuda"),
    )
