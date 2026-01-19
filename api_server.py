#!/usr/bin/env python
"""
Motherboard Fault Detection AI - API Server
REST API for processing motherboard images.
"""

import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify
import cv2
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Model paths
PATCHCORE_PATH = Path("models/patchcore.pt")
CLASSIFIER_PATH = Path("models/classifier.pt")

app = Flask(__name__)

# Global detector (loaded once at startup)
detector = None


def get_local_ip():
    """Get the local IP address for display."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def load_models():
    """Load detection models."""
    global detector

    from src.inference.predictor import load_detector_from_config

    print("Loading models...")
    detector = load_detector_from_config(
        config_path="config/config.yaml",
        patchcore_checkpoint=PATCHCORE_PATH,
        classifier_checkpoint=CLASSIFIER_PATH,
    )
    print("Models loaded successfully!")


def process_image(image_path: str) -> dict:
    """
    Process a single image and return results.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with detection results and output path
    """
    global detector

    img_path = Path(image_path)

    if not img_path.exists():
        return {
            "success": False,
            "error": f"File not found: {image_path}",
        }

    if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        return {
            "success": False,
            "error": f"Unsupported file format: {img_path.suffix}",
        }

    # Setup output directories
    output_dir = Path("inference/output")
    clean_dir = output_dir / "clean"
    fault_dir = output_dir / "fault"
    clean_dir.mkdir(parents=True, exist_ok=True)
    fault_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run prediction
        result = detector.predict(str(img_path), return_visualization=True)

        is_anomaly = result.get("is_anomaly", False)
        classification = result.get("classification", "unknown")
        confidence = result.get("confidence", 0)
        defects = result.get("defects", [])

        # Load original image
        image = cv2.imread(str(img_path))

        if is_anomaly:
            # Draw bounding boxes on faulty image
            for defect in defects:
                bbox = defect.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(c) for c in bbox]
                    # Scale bbox to original image size
                    h, w = image.shape[:2]
                    scale_x = w / 256
                    scale_y = h / 256
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    # Draw rectangle
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    # Draw label
                    label = f"{classification} ({confidence:.0%})"
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # If no bounding boxes but still anomaly, add text overlay
            if not defects:
                label = f"FAULT: {classification} ({confidence:.0%})"
                cv2.putText(image, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Create fault type subfolder
            fault_type_dir = fault_dir / classification
            fault_type_dir.mkdir(parents=True, exist_ok=True)

            # Save to fault folder
            output_path = fault_type_dir / img_path.name
            cv2.imwrite(str(output_path), image)

        else:
            # Save to clean folder
            output_path = clean_dir / img_path.name
            cv2.imwrite(str(output_path), image)

        return {
            "success": True,
            "input_path": str(img_path.absolute()),
            "output_path": str(output_path.absolute()),
            "result": "fault" if is_anomaly else "clean",
            "is_anomaly": is_anomaly,
            "classification": classification,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 1),
            "defects": defects,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


@app.route("/", methods=["GET"])
def home():
    """Health check endpoint."""
    return jsonify({
        "status": "running",
        "service": "Motherboard Fault Detection API",
        "endpoints": {
            "GET /": "Health check (this page)",
            "POST /analyze": "Analyze an image",
            "GET /status": "Check model status",
        }
    })


@app.route("/status", methods=["GET"])
def status():
    """Check model and service status."""
    return jsonify({
        "status": "running",
        "models_loaded": detector is not None,
        "patchcore_model": str(PATCHCORE_PATH.absolute()),
        "classifier_model": str(CLASSIFIER_PATH.absolute()),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze an image for motherboard faults.

    Request JSON:
        {
            "filepath": "C:/path/to/image.jpg"
        }

    Response JSON:
        {
            "success": true,
            "input_path": "...",
            "output_path": "...",
            "result": "fault" or "clean",
            "is_anomaly": true/false,
            "normal_percentage": 65.2,
            "fault_percentage": 34.8,
            "anomaly_score": 0.348,
            "classification": "burn_marks",
            "confidence": 0.87,
            "confidence_percentage": 87.0,
            "defects": [...]
        }
    """
    if detector is None:
        return jsonify({
            "success": False,
            "error": "Models not loaded",
        }), 500

    # Get filepath from request
    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "error": "No JSON data provided. Send: {\"filepath\": \"path/to/image.jpg\"}",
        }), 400

    filepath = data.get("filepath")

    if not filepath:
        return jsonify({
            "success": False,
            "error": "Missing 'filepath' in request body",
        }), 400

    # Process the image
    result = process_image(filepath)

    if result["success"]:
        return jsonify(result), 200
    else:
        return jsonify(result), 400


def ensure_models_available() -> bool:
    """Check if models exist, download if not."""
    patchcore_exists = PATCHCORE_PATH.exists()
    classifier_exists = CLASSIFIER_PATH.exists()

    if patchcore_exists and classifier_exists:
        return True

    print("\n[!] Models not found locally.")
    print(f"    PatchCore:  {'Found' if patchcore_exists else 'Missing'}")
    print(f"    Classifier: {'Found' if classifier_exists else 'Missing'}")
    print("\nAttempting to download from Hugging Face...")

    try:
        from src.utils.huggingface import ensure_models_exist
        return ensure_models_exist(PATCHCORE_PATH, CLASSIFIER_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to download models: {e}")
        return False


def main():
    """Main entry point for API server."""
    # Change to script directory
    os.chdir(Path(__file__).parent)

    # Get port from environment or use default
    port = int(os.environ.get("API_PORT", 5000))
    host = os.environ.get("API_HOST", "0.0.0.0")

    print("\n" + "=" * 60)
    print("  MOTHERBOARD FAULT DETECTION API")
    print("=" * 60)

    # Check/download models
    if not ensure_models_available():
        print("\n[ERROR] Models not available. Please train or download models first.")
        print("        Run without API mode and use options 1, 2, or 6.")
        sys.exit(1)

    # Load models
    try:
        load_models()
    except Exception as e:
        print(f"\n[ERROR] Failed to load models: {e}")
        sys.exit(1)

    # Get local IP for display
    local_ip = get_local_ip()

    print("\n" + "-" * 60)
    print("  API SERVER READY")
    print("-" * 60)
    print(f"\n  Local:   http://127.0.0.1:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print("\n  Endpoints:")
    print(f"    GET  /        - Health check")
    print(f"    GET  /status  - Model status")
    print(f"    POST /analyze - Analyze image")
    print("\n  Example usage (curl):")
    print(f'    curl -X POST http://127.0.0.1:{port}/analyze \\')
    print(f'         -H "Content-Type: application/json" \\')
    print(f'         -d "{{\\"filepath\\": \\"C:/path/to/image.jpg\\"}}"')
    print("\n  Example usage (Python):")
    print(f'    import requests')
    print(f'    r = requests.post("http://127.0.0.1:{port}/analyze",')
    print(f'                      json={{"filepath": "C:/path/to/image.jpg"}})')
    print(f'    print(r.json())')
    print("\n" + "-" * 60)
    print("  Press Ctrl+C to stop the server")
    print("-" * 60 + "\n")

    # Run Flask server
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
