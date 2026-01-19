#!/usr/bin/env python
"""
Run inference on motherboard images using the trained fault detector.
Combines PatchCore anomaly detection with EfficientNet classification.
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import cv2
import numpy as np
from tqdm import tqdm

from src.inference.predictor import FaultDetector, load_detector_from_config
from src.models.patchcore import PatchCoreModel
from src.models.classifier import DefectClassifier
from src.data.dataset import CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on motherboard images"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Directory containing images to process",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--patchcore-model",
        type=str,
        default="models/patchcore.pt",
        help="Path to trained PatchCore model",
    )
    parser.add_argument(
        "--classifier-model",
        type=str,
        default="models/classifier.pt",
        help="Path to trained classifier model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference/output",
        help="Directory for output files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly detection threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save visualization images",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results as JSON",
    )
    return parser.parse_args()


def load_models(args, config):
    """Load trained models."""
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Check model files exist
    patchcore_path = Path(args.patchcore_model)
    classifier_path = Path(args.classifier_model)

    if not patchcore_path.exists():
        print(f"Error: PatchCore model not found at {patchcore_path}")
        print("Please train the PatchCore model first:")
        print("  python scripts/train_patchcore.py")
        return None

    if not classifier_path.exists():
        print(f"Error: Classifier model not found at {classifier_path}")
        print("Please train the classifier first:")
        print("  python scripts/train_classifier.py")
        return None

    print("Loading models...")

    # Load PatchCore
    patchcore_config = config.get("patchcore", {})
    patchcore = PatchCoreModel(
        backbone=patchcore_config.get("backbone", "wide_resnet50_2"),
        layers_to_extract=patchcore_config.get("layers_to_extract", ["layer2", "layer3"]),
        num_neighbors=patchcore_config.get("num_neighbors", 9),
        input_size=tuple(patchcore_config.get("input_size", [256, 256])),
        device=device,
    )
    patchcore.load(patchcore_path)
    print(f"  PatchCore loaded from {patchcore_path}")

    # Load classifier
    classifier_config = config.get("classifier", {})
    classifier = DefectClassifier(
        backbone=classifier_config.get("backbone", "efficientnet_b0"),
        num_classes=classifier_config.get("num_classes", 5),
        pretrained=False,
    )

    checkpoint = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Classifier loaded from {classifier_path}")
    print(f"  Best validation F1: {checkpoint.get('best_f1', 'N/A')}")

    # Create detector
    inference_config = config.get("inference", {})
    fusion_config = inference_config.get("fusion", {})

    detector = FaultDetector(
        patchcore_model=patchcore,
        classifier_model=classifier,
        device=device,
        anomaly_threshold=args.threshold,
        patchcore_weight=fusion_config.get("patchcore_weight", 0.6),
        classifier_weight=fusion_config.get("classifier_weight", 0.4),
        class_names=config.get("data", {}).get("defect_classes", CLASS_NAMES),
    )

    return detector


def process_single_image(detector, image_path, args, output_dir=None):
    """Process a single image and return results."""
    image_path = Path(image_path)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": f"Could not load image: {image_path}"}

    # Run prediction
    result = detector.predict(
        image,
        return_visualization=args.save_visualization,
    )

    # Save outputs if requested
    if output_dir and args.save_visualization:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = image_path.stem

        # Save heatmap overlay
        if "visualization" in result:
            vis_path = output_dir / f"{base_name}_heatmap.png"
            cv2.imwrite(
                str(vis_path),
                cv2.cvtColor(result["visualization"]["heatmap_overlay"], cv2.COLOR_RGB2BGR)
            )

            # Save image with boxes
            if result["defects"]:
                boxes_path = output_dir / f"{base_name}_boxes.png"
                cv2.imwrite(
                    str(boxes_path),
                    cv2.cvtColor(result["visualization"]["image_with_boxes"], cv2.COLOR_RGB2BGR)
                )

        # Save heatmaps
        from src.visualization.heatmap import save_heatmap
        heatmap_path = output_dir / f"{base_name}_combined_heatmap.png"
        save_heatmap(result["heatmaps"]["combined"], heatmap_path)

        result["heatmap_path"] = str(heatmap_path)

    # Clean up heavy data for JSON output
    result_clean = {k: v for k, v in result.items()
                    if k not in ["heatmaps", "visualization"]}

    return result_clean


def print_result(image_name, result):
    """Pretty print a single result."""
    print(f"\n{'=' * 50}")
    print(f"Image: {image_name}")
    print(f"{'=' * 50}")

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    status = "DEFECT DETECTED" if result["is_anomaly"] else "NORMAL"
    print(f"Status: {status}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")

    if result["is_anomaly"]:
        print(f"Classification: {result['classification']} ({result['confidence']:.2%})")

        if result["defects"]:
            print(f"\nDetected Defects ({len(result['defects'])}):")
            for i, defect in enumerate(result["defects"], 1):
                bbox = defect["bbox"]
                print(f"  {i}. {defect['class']} at [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] "
                      f"(conf: {defect['confidence']:.2%})")

    print(f"\nClass Probabilities:")
    for class_name, prob in result.get("class_probabilities", {}).items():
        bar = "█" * int(prob * 20)
        print(f"  {class_name:20s}: {bar} {prob:.2%}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Motherboard Fault Detector - Inference")
    print("=" * 60)

    # Check inputs
    if not args.image and not args.image_dir:
        print("Error: Please specify --image or --image-dir")
        return 1

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Load models
    detector = load_models(args, config)
    if detector is None:
        return 1

    # Process images
    results = {}

    if args.image:
        # Single image
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return 1

        result = process_single_image(
            detector,
            image_path,
            args,
            output_dir=args.output_dir if args.save_visualization else None,
        )
        results[str(image_path)] = result

        if not args.json_output:
            print_result(image_path.name, result)

    elif args.image_dir:
        # Directory of images
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"Error: Directory not found: {image_dir}")
            return 1

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = [
            f for f in image_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"No images found in {image_dir}")
            return 1

        print(f"\nProcessing {len(image_files)} images...")

        for image_path in tqdm(image_files, desc="Processing"):
            result = process_single_image(
                detector,
                image_path,
                args,
                output_dir=args.output_dir if args.save_visualization else None,
            )
            results[str(image_path)] = result

        # Print summary
        if not args.json_output:
            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)

            total = len(results)
            anomalies = sum(1 for r in results.values()
                          if r.get("is_anomaly", False))
            errors = sum(1 for r in results.values()
                        if "error" in r)

            print(f"Total images: {total}")
            print(f"Anomalies detected: {anomalies}")
            print(f"Normal: {total - anomalies - errors}")
            print(f"Errors: {errors}")

            if anomalies > 0:
                print("\nDefect type distribution:")
                class_counts = {}
                for r in results.values():
                    if r.get("is_anomaly"):
                        cls = r.get("classification", "unknown")
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                    print(f"  {cls}: {count}")

    # JSON output
    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        # Save results to JSON file
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

        if args.save_visualization:
            print(f"Visualizations saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
