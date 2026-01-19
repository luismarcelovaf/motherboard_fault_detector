"""
Image preprocessing pipeline for motherboard fault detection.
Includes color normalization, CLAHE enhancement, and noise reduction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path


class ImagePreprocessor:
    """
    Preprocessor for motherboard images with CLAHE enhancement and denoising.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        apply_denoise: bool = True,
        denoise_h: int = 10,
        denoise_template_window_size: int = 7,
        denoise_search_window_size: int = 21,
    ):
        """
        Initialize the preprocessor.

        Args:
            target_size: Output image size (height, width)
            apply_clahe: Whether to apply CLAHE enhancement
            clahe_clip_limit: CLAHE clip limit for contrast limiting
            clahe_tile_grid_size: CLAHE tile grid size
            apply_denoise: Whether to apply denoising
            denoise_h: Denoising filter strength
            denoise_template_window_size: Template window size for denoising
            denoise_search_window_size: Search window size for denoising
        """
        self.target_size = target_size
        self.apply_clahe = apply_clahe
        self.apply_denoise = apply_denoise

        # CLAHE parameters
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        # Denoising parameters
        self.denoise_h = denoise_h
        self.denoise_template_window_size = denoise_template_window_size
        self.denoise_search_window_size = denoise_search_window_size

        # Create CLAHE object
        if self.apply_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size
            )

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file.

        Args:
            image_path: Path to the image file

        Returns:
            Image as numpy array in BGR format
        """
        image_path = str(image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image

    def resize(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            image: Input image
            size: Target size (height, width), uses self.target_size if None

        Returns:
            Resized image
        """
        if size is None:
            size = self.target_size
        return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    def apply_clahe_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input BGR image

        Returns:
            Enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)

        # Merge channels back
        lab = cv2.merge([l_channel, a_channel, b_channel])

        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply non-local means denoising.

        Args:
            image: Input BGR image

        Returns:
            Denoised image
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            self.denoise_h,
            self.denoise_h,
            self.denoise_template_window_size,
            self.denoise_search_window_size
        )

    def normalize_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize color distribution using histogram equalization per channel.

        Args:
            image: Input BGR image

        Returns:
            Color normalized image
        """
        # Split channels
        b, g, r = cv2.split(image)

        # Equalize each channel
        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)

        return cv2.merge([b, g, r])

    def preprocess(
        self,
        image: np.ndarray,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline.

        Args:
            image: Input BGR image
            size: Target size (height, width), uses self.target_size if None

        Returns:
            Preprocessed image
        """
        # Resize first
        image = self.resize(image, size)

        # Apply CLAHE enhancement
        if self.apply_clahe:
            image = self.apply_clahe_enhancement(image)

        # Apply denoising
        if self.apply_denoise:
            image = self.denoise(image)

        return image

    def preprocess_for_model(
        self,
        image: np.ndarray,
        size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image: Input BGR image
            size: Target size (height, width)
            normalize: Whether to normalize to [0, 1] range

        Returns:
            Preprocessed image ready for model input (RGB, normalized)
        """
        # Apply preprocessing
        image = self.preprocess(image, size)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        if normalize:
            image = image.astype(np.float32) / 255.0

        return image

    def process_file(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Load, preprocess, and optionally save an image.

        Args:
            image_path: Path to input image
            output_path: Path to save preprocessed image (optional)
            size: Target size (height, width)

        Returns:
            Preprocessed image
        """
        # Load image
        image = self.load_image(image_path)

        # Preprocess
        processed = self.preprocess(image, size)

        # Save if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed)

        return processed


def create_preprocessor_from_config(config: dict) -> ImagePreprocessor:
    """
    Create an ImagePreprocessor from a configuration dictionary.

    Args:
        config: Configuration dictionary with preprocessing parameters

    Returns:
        Configured ImagePreprocessor instance
    """
    preproc_config = config.get("preprocessing", {})

    return ImagePreprocessor(
        target_size=tuple(preproc_config.get("target_size", [224, 224])),
        apply_clahe=True,
        clahe_clip_limit=preproc_config.get("clahe", {}).get("clip_limit", 2.0),
        clahe_tile_grid_size=tuple(
            preproc_config.get("clahe", {}).get("tile_grid_size", [8, 8])
        ),
        apply_denoise=True,
        denoise_h=preproc_config.get("denoise", {}).get("h", 10),
        denoise_template_window_size=preproc_config.get("denoise", {}).get(
            "template_window_size", 7
        ),
        denoise_search_window_size=preproc_config.get("denoise", {}).get(
            "search_window_size", 21
        ),
    )
