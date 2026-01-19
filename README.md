# Motherboard Fault Detection AI

An AI system for detecting faulty and tampered Dell XPS motherboards using computer vision and deep learning.

## Features

- **Anomaly Detection**: Identifies if a motherboard has any defects (PatchCore algorithm)
- **Defect Classification**: Categorizes the type of defect (EfficientNet classifier)
- **Defect Localization**: Shows WHERE the defect is located (Grad-CAM++ heatmaps)
- **Small Dataset Optimized**: Works with as few as 20-30 images using heavy augmentation

## Defect Categories

| Category | Description |
|----------|-------------|
| `burn_marks` | Electrical damage, charring, overheating |
| `reuse_marks` | Scratches, wear patterns, handling marks |
| `liquid_damage` | Water/chemical corrosion, white residue |
| `label_tampering` | Missing or altered serial number labels |
| `other_tampering` | Miscellaneous physical tampering |
| `blown_capacitors` | Bulging, leaking, or exploded capacitors |
| `oxidation_damage` | Rust, green corrosion, oxidation near ICs |

## Quick Start

### Windows (Recommended)

Simply double-click `run.bat` or run from command line:

```bash
cd motherboard_fault_detector
run.bat
```

The script will automatically:
- Check for Python 3.10/3.11/3.12
- Create a virtual environment if needed
- Install PyTorch with CUDA support (falls back to CPU if needed)
- Install all dependencies
- Launch the interactive menu

### Manual Setup

If you prefer manual setup:

```bash
cd motherboard_fault_detector
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python main.py
```

This will show you options to:
- Train the anomaly detector
- Train the classifier
- Run inference on images
- Check dataset status
- Download/upload models from Hugging Face

### Using Pre-trained Models

Models are stored on Hugging Face Hub to avoid bloating the git repository.

**Download models (no training needed):**
- Select option [7] in the menu, or
- Models auto-download when you try to run inference

**Upload your trained models:**
- Select option [8] in the menu
- Requires Hugging Face login: `huggingface-cli login`

**Custom repository:**
Set `HF_REPO_ID` environment variable to use your own repository:
```bash
set HF_REPO_ID=your-username/your-repo-name
```

### Or Run Scripts Directly

```bash
# Train PatchCore (anomaly detection)
python scripts/train_patchcore.py --data-dir data/raw/normal

# Train classifier (defect classification)
python scripts/train_classifier.py --data-dir data/raw

# Run inference
python scripts/inference.py --image path/to/image.jpg --save-visualization
```

## Project Structure

```
motherboard_fault_detector/
├── run.bat                 # Entry point (START HERE)
├── main.py                 # Interactive menu
├── requirements.txt        # Python dependencies
├── config/
│   └── config.yaml         # All configuration parameters
├── data/
│   └── raw/                # Training images
│       ├── normal/         # Clean motherboard images
│       ├── burn_marks/     # Burn damage images
│       ├── liquid_damage/  # Corrosion/water damage
│       └── ...             # Other defect categories
├── inference/
│   ├── input/              # Place images here for analysis
│   └── output/
│       ├── clean/          # Images detected as clean
│       └── fault/          # Images with defects (by type)
│           ├── burn_marks/
│           ├── liquid_damage/
│           └── ...
├── src/
│   ├── data/               # Data loading & augmentation
│   ├── models/             # PatchCore & EfficientNet
│   ├── inference/          # Prediction pipeline
│   ├── utils/              # Hugging Face integration
│   └── visualization/      # Grad-CAM & heatmaps
├── scripts/
│   ├── train_patchcore.py  # Train anomaly detector
│   ├── train_classifier.py # Train defect classifier
│   └── inference.py        # Run predictions
└── outputs/
    └── models/             # Saved model weights (on HuggingFace)
```

## Running Inference

1. Place images to analyze in `inference/input/`
2. Run `run.bat` and select option [3]
3. Choose which image(s) to process
4. Results are saved to:
   - `inference/output/clean/` - Clean motherboards
   - `inference/output/fault/<defect_type>/` - Faulty boards with bounding boxes

## How It Works

### Two-Stage Detection Pipeline

```
Input Image
     │
     ▼
┌─────────────────────────────────────┐
│  Stage 1: ANOMALY DETECTION         │
│  (PatchCore)                        │
│                                     │
│  - Compares image to "normal" bank  │
│  - Outputs: Anomaly Score (0-1)     │
│  - Score > 0.5 = Defect detected    │
└─────────────────────────────────────┘
     │
     ▼ (if anomaly detected)
┌─────────────────────────────────────┐
│  Stage 2: CLASSIFICATION            │
│  (EfficientNet)                     │
│                                     │
│  - Identifies defect TYPE           │
│  - Outputs: Class + Confidence      │
│  - Grad-CAM shows defect LOCATION   │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  OUTPUT                             │
│                                     │
│  {                                  │
│    "is_anomaly": true,              │
│    "anomaly_score": 0.87,           │
│    "classification": "burn_marks",  │
│    "confidence": 0.92,              │
│    "defects": [{bbox, type}]        │
│  }                                  │
└─────────────────────────────────────┘
```

### Why Two Stages?

1. **PatchCore** excels at detecting ANY anomaly with very few training samples (only needs normal images)
2. **EfficientNet** provides specific defect categorization when trained on labeled defect images
3. **Combined approach** gives both high detection rate AND actionable classification

## Training Data Requirements

### Minimum Recommended:
- **Normal images**: 10-20 clean motherboard photos
- **Defect images**: 3-5 per category (augmented to 150-250 each)

### Current Dataset:
```
normal:           8 images
burn_marks:       4 images
liquid_damage:    4 images
label_tampering:  2 images
other_tampering:  2 images
reuse_marks:      1 image
blown_capacitors: 1 image
oxidation_damage: 2 images
```

### Adding More Images:
Simply drop new images into the appropriate `data/raw/<category>/` folder.

## Configuration

Edit `config/config.yaml` to adjust:

```yaml
# Anomaly detection threshold
patchcore:
  thresholds:
    anomaly_score: 0.5  # Lower = more sensitive

# Training parameters
classifier:
  training:
    epochs: 100
    batch_size: 16
    learning_rate: 0.001

# Data augmentation
augmentation:
  defect_expansion_factor: 50  # Each image → 50 augmented versions
```

## Output Format

Inference produces JSON results:

```json
{
  "is_anomaly": true,
  "anomaly_score": 0.87,
  "classification": "burn_marks",
  "confidence": 0.92,
  "class_probabilities": {
    "burn_marks": 0.92,
    "liquid_damage": 0.05,
    "oxidation_damage": 0.02,
    ...
  },
  "defects": [
    {
      "bbox": [120, 80, 250, 180],
      "confidence": 0.89,
      "class": "burn_marks"
    }
  ]
}
```

## Hardware Requirements

- **Minimum**: CPU with 8GB RAM (slow training)
- **Recommended**: NVIDIA GPU with 4GB+ VRAM (10-50x faster)

The system automatically detects and uses GPU if available.

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch-size 8` or `--batch-size 4`

### "No module named X"
- Run: `pip install -r requirements.txt`

### Poor detection accuracy
- Add more training images
- Ensure images are properly cropped (motherboard only, no backgrounds)
- Check that normal images are truly defect-free

## License

For internal use only. Training images may be subject to copyright.

## Credits

- **PatchCore**: Intel's anomalib library
- **EfficientNet**: Google's efficient CNN architecture
- **Grad-CAM++**: Visualization technique for CNN attention
