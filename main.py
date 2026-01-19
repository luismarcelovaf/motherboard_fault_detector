#!/usr/bin/env python
"""
Motherboard Fault Detection AI - Main Entry Point
Interactive menu for training and inference.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Model paths
PATCHCORE_PATH = Path("outputs/models/patchcore.pt")
CLASSIFIER_PATH = Path("outputs/models/classifier.pt")


def print_header():
    """Print application header."""
    print("\n" + "=" * 60)
    print("  MOTHERBOARD FAULT DETECTION AI")
    print("  Dell XPS Motherboard Defect Detection System")
    print("=" * 60)


def print_menu():
    """Print main menu options."""
    print("\nWhat would you like to do?\n")
    print("  [1] Train PatchCore (Anomaly Detection)")
    print("      - Builds memory bank from normal images")
    print("      - Only needs normal/clean motherboard images")
    print()
    print("  [2] Train Classifier (Defect Classification)")
    print("      - Trains EfficientNet to identify defect types")
    print("      - Uses augmentation to expand small dataset")
    print()
    print("  [3] Run Inference")
    print("      - Analyze images from inference/input folder")
    print("      - Results saved to inference/output/clean or /fault")
    print()
    print("  [4] Show Dataset Status")
    print("      - Display current training data statistics")
    print()
    print("  [5] Quick Test (Check Installation)")
    print("      - Verify all dependencies are installed")
    print()
    print("  [6] Download Models from Hugging Face")
    print("      - Download pre-trained models if available")
    print()
    print("  [7] Upload Models to Hugging Face")
    print("      - Share your trained models")
    print()
    print("  [0] Exit")
    print()


def check_dependencies():
    """Check if all required packages are installed."""
    print("\nChecking dependencies...")

    required = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "Timm (pretrained models)"),
        ("cv2", "OpenCV"),
        ("albumentations", "Albumentations"),
        ("sklearn", "Scikit-learn"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("huggingface_hub", "Hugging Face Hub"),
    ]

    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name}")
            missing.append(name)

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("  [WARN] CUDA not available - will use CPU (slower)")
    except:
        pass

    # Check model status
    print("\nModel Status:")
    print(f"  PatchCore:  {'[OK] Found' if PATCHCORE_PATH.exists() else '[--] Not found'}")
    print(f"  Classifier: {'[OK] Found' if CLASSIFIER_PATH.exists() else '[--] Not found'}")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies installed!")
        return True


def show_dataset_status():
    """Show current dataset statistics."""
    data_dir = Path("data/raw")

    print("\n" + "-" * 40)
    print("DATASET STATUS")
    print("-" * 40)

    if not data_dir.exists():
        print("Data directory not found!")
        print(f"Expected: {data_dir.absolute()}")
        return

    total_normal = 0
    total_defects = 0

    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir():
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            count = len(images)

            if folder.name == "normal":
                total_normal = count
                print(f"  {folder.name:20s}: {count:3d} images  [NORMAL]")
            else:
                total_defects += count
                status = "[OK]" if count >= 3 else "[LOW]"
                print(f"  {folder.name:20s}: {count:3d} images  {status}")

    print("-" * 40)
    print(f"  {'TOTAL':20s}: {total_normal + total_defects:3d} images")
    print(f"  Normal: {total_normal}, Defects: {total_defects}")

    if total_normal < 5:
        print("\n  [!] Warning: Need more normal images for PatchCore")
    if total_defects < 10:
        print("  [!] Warning: Small defect dataset - heavy augmentation will be used")


def prompt_upload_model(model_type: str, model_path: Path):
    """Prompt user to upload model after training."""
    if not model_path.exists():
        return

    print("\n" + "-" * 40)
    upload = input(f"Upload {model_type} model to Hugging Face? [y/N]: ").strip().lower()
    if upload == 'y':
        try:
            from src.utils.huggingface import upload_model
            upload_model(model_path, model_type)
        except ImportError:
            print("[ERROR] Hugging Face Hub not installed. Run: pip install huggingface_hub")
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")


def train_patchcore():
    """Run PatchCore training."""
    print("\n" + "-" * 40)
    print("TRAINING PATCHCORE")
    print("-" * 40)

    data_dir = Path("data/raw/normal")
    if not data_dir.exists() or len(list(data_dir.glob("*"))) == 0:
        print("Error: No normal images found in data/raw/normal/")
        print("Please add clean motherboard images first.")
        return

    print(f"Normal images directory: {data_dir}")
    print(f"Found {len(list(data_dir.glob('*')))} images")

    confirm = input("\nStart training? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # Run training script
    import subprocess
    result = subprocess.run([
        sys.executable, "scripts/train_patchcore.py",
        "--data-dir", str(data_dir),
        "--output", str(PATCHCORE_PATH)
    ])

    if result.returncode == 0:
        prompt_upload_model("patchcore", PATCHCORE_PATH)


def train_classifier():
    """Run classifier training."""
    print("\n" + "-" * 40)
    print("TRAINING CLASSIFIER")
    print("-" * 40)

    data_dir = Path("data/raw")
    if not data_dir.exists():
        print("Error: Data directory not found!")
        return

    # Count defect images
    defect_count = 0
    for folder in data_dir.iterdir():
        if folder.is_dir() and folder.name != "normal":
            defect_count += len(list(folder.glob("*")))

    print(f"Data directory: {data_dir}")
    print(f"Found {defect_count} defect images")

    if defect_count < 5:
        print("\nWarning: Very few defect images. Results may be poor.")

    # Get training parameters
    print("\nTraining parameters:")
    epochs = input("  Epochs [100]: ").strip() or "100"
    batch_size = input("  Batch size [16]: ").strip() or "16"

    confirm = input("\nStart training? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # Run training script
    import subprocess
    result = subprocess.run([
        sys.executable, "scripts/train_classifier.py",
        "--data-dir", str(data_dir),
        "--epochs", epochs,
        "--batch-size", batch_size,
        "--output", str(CLASSIFIER_PATH)
    ])

    if result.returncode == 0:
        prompt_upload_model("classifier", CLASSIFIER_PATH)


def ensure_models_available() -> bool:
    """Check if models exist, offer to download if not."""
    patchcore_exists = PATCHCORE_PATH.exists()
    classifier_exists = CLASSIFIER_PATH.exists()

    if patchcore_exists and classifier_exists:
        return True

    print("\n[!] Models not found locally.")
    print(f"    PatchCore:  {'Found' if patchcore_exists else 'Missing'}")
    print(f"    Classifier: {'Found' if classifier_exists else 'Missing'}")

    download = input("\nAttempt to download from Hugging Face? [Y/n]: ").strip().lower()
    if download == 'n':
        print("\nPlease train the models first (options 1 and 2).")
        return False

    try:
        from src.utils.huggingface import ensure_models_exist
        return ensure_models_exist(PATCHCORE_PATH, CLASSIFIER_PATH)
    except ImportError:
        print("[ERROR] Hugging Face Hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("\nPlease train the models first (options 1 and 2).")
        return False


def run_inference():
    """Run inference on images from inference/input folder."""
    print("\n" + "-" * 40)
    print("INFERENCE")
    print("-" * 40)

    # Check/download models
    if not ensure_models_available():
        return

    # Setup paths
    input_dir = Path("inference/input")
    output_dir = Path("inference/output")
    clean_dir = output_dir / "clean"
    fault_dir = output_dir / "fault"

    # Create input directory if needed
    input_dir.mkdir(parents=True, exist_ok=True)

    # Clean up output directories
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    fault_dir.mkdir(parents=True, exist_ok=True)

    # Get list of images
    images = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg")))

    if len(images) == 0:
        print(f"\nNo images found in {input_dir}")
        print("Please add images to the inference/input folder.")
        return

    # Display image list
    print(f"\nFound {len(images)} images in {input_dir}:\n")
    print("  [0] Process ALL images")
    for i, img in enumerate(images, 1):
        print(f"  [{i}] {img.name}")
    print()

    # Get user selection
    choice = input(f"Select image [0-{len(images)}]: ").strip()

    try:
        choice_num = int(choice)
        if choice_num < 0 or choice_num > len(images):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return

    # Determine which images to process
    if choice_num == 0:
        images_to_process = images
        print(f"\nProcessing all {len(images)} images...")
    else:
        images_to_process = [images[choice_num - 1]]
        print(f"\nProcessing: {images_to_process[0].name}")

    # Run inference
    process_images_inference(images_to_process, clean_dir, fault_dir)


def process_images_inference(images: list, clean_dir: Path, fault_dir: Path):
    """Process images and save results to appropriate folders."""
    import cv2
    import json

    # Import inference components
    try:
        from src.inference.predictor import load_detector_from_config
    except ImportError as e:
        print(f"[ERROR] Could not import inference module: {e}")
        return

    print("\nLoading models...")
    try:
        detector = load_detector_from_config(
            config_path="config/config.yaml",
            patchcore_checkpoint=PATCHCORE_PATH,
            classifier_checkpoint=CLASSIFIER_PATH,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return

    print("Models loaded.\n")

    results_summary = []

    for img_path in images:
        print(f"Processing: {img_path.name}...", end=" ")

        try:
            # Run prediction
            result = detector.predict(str(img_path), return_visualization=True)

            is_anomaly = result.get("is_anomaly", False)
            classification = result.get("classification", "unknown")
            confidence = result.get("confidence", 0)
            anomaly_score = result.get("anomaly_score", 0)

            # Load original image
            image = cv2.imread(str(img_path))

            if is_anomaly:
                # Draw bounding boxes on faulty image
                defects = result.get("defects", [])
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
                print(f"FAULT ({classification}, {confidence:.0%}) -> {output_path}")

            else:
                # Save to clean folder
                output_path = clean_dir / img_path.name
                cv2.imwrite(str(output_path), image)
                print(f"CLEAN (score: {anomaly_score:.2f}) -> {output_path}")

            results_summary.append({
                "image": img_path.name,
                "is_anomaly": is_anomaly,
                "classification": classification,
                "confidence": confidence,
                "anomaly_score": anomaly_score,
                "output": str(output_path),
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results_summary.append({
                "image": img_path.name,
                "error": str(e),
            })

    # Print summary
    print("\n" + "=" * 50)
    print("INFERENCE SUMMARY")
    print("=" * 50)

    clean_count = sum(1 for r in results_summary if not r.get("is_anomaly", True) and "error" not in r)
    fault_count = sum(1 for r in results_summary if r.get("is_anomaly", False) and "error" not in r)
    error_count = sum(1 for r in results_summary if "error" in r)

    print(f"  Clean:  {clean_count}")
    print(f"  Faulty: {fault_count}")
    if error_count > 0:
        print(f"  Errors: {error_count}")
    print(f"\nResults saved to: inference/output/")


def download_models():
    """Download models from Hugging Face."""
    print("\n" + "-" * 40)
    print("DOWNLOAD MODELS FROM HUGGING FACE")
    print("-" * 40)

    try:
        from src.utils.huggingface import download_model, get_repo_id, check_remote_models

        repo_id = get_repo_id()
        print(f"\nRepository: {repo_id}")
        print("\nChecking available models...")

        remote_status = check_remote_models()
        print(f"  PatchCore:  {'Available' if remote_status.get('patchcore') else 'Not found'}")
        print(f"  Classifier: {'Available' if remote_status.get('classifier') else 'Not found'}")

        if not any(remote_status.values()):
            print("\n[!] No models found in repository.")
            print("    You may need to train and upload models first.")
            return

        # Download PatchCore
        if remote_status.get('patchcore'):
            if PATCHCORE_PATH.exists():
                overwrite = input("\nPatchCore model exists locally. Overwrite? [y/N]: ").strip().lower()
                if overwrite != 'y':
                    print("Skipping PatchCore download.")
                else:
                    download_model(PATCHCORE_PATH, "patchcore")
            else:
                download_model(PATCHCORE_PATH, "patchcore")

        # Download Classifier
        if remote_status.get('classifier'):
            if CLASSIFIER_PATH.exists():
                overwrite = input("\nClassifier model exists locally. Overwrite? [y/N]: ").strip().lower()
                if overwrite != 'y':
                    print("Skipping Classifier download.")
                else:
                    download_model(CLASSIFIER_PATH, "classifier")
            else:
                download_model(CLASSIFIER_PATH, "classifier")

        print("\nDownload complete!")

    except ImportError:
        print("[ERROR] Hugging Face Hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")


def upload_models():
    """Upload models to Hugging Face."""
    print("\n" + "-" * 40)
    print("UPLOAD MODELS TO HUGGING FACE")
    print("-" * 40)

    try:
        from src.utils.huggingface import upload_model, get_repo_id, ensure_login

        repo_id = get_repo_id()
        print(f"\nRepository: {repo_id}")
        print(f"\nNote: Set HF_REPO_ID environment variable to use a different repository.")

        # Check local models
        print("\nLocal models:")
        print(f"  PatchCore:  {'Found' if PATCHCORE_PATH.exists() else 'Not found'}")
        print(f"  Classifier: {'Found' if CLASSIFIER_PATH.exists() else 'Not found'}")

        if not PATCHCORE_PATH.exists() and not CLASSIFIER_PATH.exists():
            print("\n[!] No models found. Please train models first.")
            return

        # Ensure login
        if not ensure_login():
            return

        # Upload PatchCore
        if PATCHCORE_PATH.exists():
            upload_pc = input("\nUpload PatchCore model? [Y/n]: ").strip().lower()
            if upload_pc != 'n':
                upload_model(PATCHCORE_PATH, "patchcore")

        # Upload Classifier
        if CLASSIFIER_PATH.exists():
            upload_cl = input("\nUpload Classifier model? [Y/n]: ").strip().lower()
            if upload_cl != 'n':
                upload_model(CLASSIFIER_PATH, "classifier")

        print("\nUpload complete!")

    except ImportError:
        print("[ERROR] Hugging Face Hub not installed. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")


def main():
    """Main entry point."""
    # Change to script directory
    os.chdir(Path(__file__).parent)

    print_header()

    while True:
        print_menu()

        choice = input("Enter choice [0-7]: ").strip()

        if choice == "0":
            print("\nGoodbye!")
            break
        elif choice == "1":
            train_patchcore()
        elif choice == "2":
            train_classifier()
        elif choice == "3":
            run_inference()
        elif choice == "4":
            show_dataset_status()
        elif choice == "5":
            check_dependencies()
        elif choice == "6":
            download_models()
        elif choice == "7":
            upload_models()
        else:
            print("\nInvalid choice. Please enter 0-7.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
