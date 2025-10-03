# Multi-QR Code Recognition for Medicine Packs - Stage 1: Detection

This repository contains a complete solution for **Stage 1: QR Code Detection** on medicine packs using YOLOv12. Our solution detects multiple QR codes (manufacturer, batch number, distributor, regulator) from medicine pack images without implementing the bonus Stage 2 decoding task.

## Solution Overview

Our approach focuses on **Stage 1: Detection Only**:

- **YOLOv12 Small** for robust QR code detection
- **Hybrid annotation strategy** using OpenCV detection + Label Studio manual labeling
- **Kaggle training** for GPU acceleration
- **Production-ready inference pipeline** for detection
- **No Stage 2 implementation** (bonus decoding/classification not attempted)

### Key Features - Stage 1: Detection

- Detects multiple QR codes per image
- Handles tilted, blurred, and partially covered QR codes
- Trained on 200 medicine pack images
- Outputs `submission_detection_1.json` in required format
- Complete reproducible detection pipeline
- **Stage 2 (bonus decoding) not implemented**

## Repository Structure

```
multiqr-hackathon/
│
├── README.md                # This file - Setup & usage instructions
├── requirements.txt         # Python dependencies
├── train.py                 # Training script
├── infer.py                 # Inference script (input=images → output=JSON)
├── evaluate.py              # Evaluation script for self-check
│
├── data/                    # Dataset placeholder
│   └── demo_images/         # Demo images for testing
│
├── outputs/
│   ├── submission_detection_1.json   # Required output file (Stage 1)
│   └── best.pt              # Trained model weights
│
└── src/                     # Source code modules
    ├── models/
    │   └── yolo_detector.py # YOLO model wrapper
    ├── datasets/
    │   └── preprocessing.py # Data preprocessing utilities
    ├── utils/
    │   └── evaluation.py    # Evaluation utilities
    └── __init__.py
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd multiqr-hackathon

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download and extract the QR_Dataset to data/ folder
# Your structure should look like:
# data/
#   QR_Dataset/
#     train_images/          # 200 training images (img001.jpg to img200.jpg)
#     test_images/           # 50 test images (img201.jpg to img250.jpg)
#   combined.csv             # Annotations file (1024 QR code bounding boxes)
#   demo_images/             # Sample images for testing
```

**Note**: The `combined.csv` file contains our hybrid annotations combining:

- OpenCV automated detection results
- Label Studio manual annotations for complex cases
- Format: `image_id,x_min,y_min,x_max,y_max`

### 3. Training (Kaggle - Recommended)

Our model was trained on Kaggle for GPU acceleration:

```bash
# Model was trained on Kaggle using:
# - YOLOv12 Small (yolo12s.pt)
# - 50 epochs with patience=100
# - Batch size 16, Image size 640x640
# - Achieved 99.16% F1 Score

# For local training (slower):
python train.py --data-dir data/QR_Dataset --annotations data/combined.csv --epochs 50 --gpu

# Training outputs are available in outputs/best.pt
```

### 4. Inference

```bash
# Run inference on test images (Stage 1: Detection)
python infer.py --input data/QR_Dataset/test_images --output outputs/submission_detection_1.json

# With custom model and visualization
python infer.py --input data/demo_images --model outputs/best.pt --conf 0.25 --visualize
```

### 5. Evaluation (Optional)

```bash
# Evaluate predictions against ground truth
python evaluate.py --predictions outputs/submission_detection_1.json --ground-truth data/combined.csv --detailed-report --visualize --images-dir data/QR_Dataset/test_images
```

## Configuration Options

### Training Parameters

```bash
python train.py \
  --data-dir data/QR_Dataset \
  --annotations data/combined.csv \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --gpu \
  --name qr_detection_v2
```

### Inference Parameters

```bash
python infer.py \
  --input data/test_images \
  --output outputs/submission.json \
  --model outputs/best.pt \
  --conf 0.25 \
  --iou 0.45 \
  --visualize \
  --ground-truth data/combined.csv
```

## Training on Kaggle

For GPU acceleration, we recommend training on Kaggle:

### 1. Upload Data to Kaggle

- Upload QR_Dataset as a Kaggle dataset
- Upload combined.csv as input

### 2. Use Kaggle Training Mode

```python
# In Kaggle notebook or with Kaggle paths
python train.py --kaggle --epochs 50
```

### 3. Download Trained Model

- Download the trained model weights (`best.pt`) from Kaggle output
- Place in `outputs/best.pt` for local inference

## Model Performance

Our trained model achieves **excellent performance** on the QR detection task:

### Validation Metrics (Actual Results)

- **mAP50-95**: 0.6074 (Mean Average Precision across IoU thresholds)
- **mAP50**: 0.9942 (Mean Average Precision at IoU=0.5)
- **mAP75**: 0.6590 (Mean Average Precision at IoU=0.75)
- **Precision**: 0.9889 (98.89% - Very low false positives)
- **Recall**: 0.9942 (99.42% - Excellent detection rate)
- **F1 Score**: 0.9916 (99.16% - Outstanding overall performance)
- **Accuracy**: 0.9624 (96.24% - High classification accuracy)

### Performance Analysis

- **Outstanding Detection**: 99.42% recall means we detect almost all QR codes
- **High Precision**: 98.89% precision means very few false detections
- **Fast Inference**: 16.6ms per image processing time
- **Robust Performance**: Excellent performance across different IoU thresholds

### Training Details

- **Model**: YOLOv12 Small (yolo12s.pt) - Trained on Kaggle
- **Platform**: Kaggle T4 GPU for accelerated training
- **Input Size**: 640x640 pixels
- **Training Images**: 200 (img001.jpg to img200.jpg)
- **Validation Images**: 50 (img201.jpg to img250.jpg)
- **Epochs**: 50 with patience=100 (early stopping)
- **Batch Size**: 16
- **Data Augmentation**: Built-in YOLO augmentations

## Output Format

The inference script generates `submission_detection_1.json` in the required format:

```json
[
  {
    "image_id": "img001.jpg",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  },
  {
    "image_id": "img002.jpg",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]
```

## Data Processing Pipeline

### 1. Multi-Source Annotation Workflow

**Our hybrid annotation approach combined automated and manual labeling:**

1. **OpenCV QRCodeDetector**:

   - Automated detection on clear, well-positioned QR codes
   - Generated initial bounding boxes for obvious cases
   - Used `cv2classification.ipynb` for bulk processing

2. **Label Studio Manual Annotation**:

   - Manual annotation for complex cases (tilted, blurred, partially covered)
   - Used Label Studio web interface for precise bounding box creation
   - Handled edge cases that OpenCV missed

3. **Combined Dataset Creation**:
   - Merged OpenCV and Label Studio annotations using `combinedcsv.ipynb`
   - Removed false positives from OpenCV detection
   - Eliminated duplicate annotations
   - **Final output**: `data/combined.csv` with 1024 total annotations

### 2. YOLO Format Conversion

- Bounding boxes converted from (x_min, y_min, x_max, y_max) to YOLO format (normalized)
- Dataset split: train (img001-img200), test (img201-img250)
- Data augmentation handled by YOLO training pipeline

### 3. Quality Assurance

- False positive removal from OpenCV detection (removed problematic images)
- Annotation validation and format checking
- Comprehensive evaluation metrics for model performance

## Technical Implementation

### Key Technologies

- **YOLOv12**: Latest state-of-the-art object detection model
- **Ultralytics**: Production-ready YOLO implementation
- **OpenCV**: Image processing and QR detection
- **pandas**: Data manipulation and CSV handling
- **Kaggle**: GPU-accelerated training platform

### Model Architecture

```python
# YOLOv12 Small Configuration (Trained Model)
model = YOLO('yolo12s.pt')  # YOLOv12 Small pretrained base model
# Single class: QR code (class 0)
# Input: 640x640 RGB images
# Output: Bounding boxes with confidence scores
# Performance: 16.6ms inference time per image
```

## Performance Optimizations

1. **Efficient Model Choice**: YOLOv11 Nano for speed vs accuracy balance
2. **Optimal Input Size**: 640x640 for medicine pack images
3. **Smart Augmentation**: Built-in YOLO augmentations
4. **Early Stopping**: Prevents overfitting on small dataset
5. **TTA (Test Time Augmentation)**: Can be enabled for better accuracy

## Common Issues & Solutions

### Training Issues

```bash
# GPU memory issues
python train.py --batch 8  # Reduce batch size

# Slow training
python train.py --workers 0  # Disable multiprocessing

# Overfitting
python train.py --patience 20  # Enable early stopping
```

### Inference Issues

```bash
# Low detection rate
python infer.py --conf 0.15  # Lower confidence threshold

# Too many false positives
python infer.py --conf 0.4   # Higher confidence threshold

# Missing model file
# Download best.pt from Kaggle training output
```

## Evaluation Metrics

The evaluation script provides comprehensive metrics:

### Detection Metrics

- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of Precision and Recall
- **mAP**: Mean Average Precision at IoU thresholds

### Per-Image Analysis

- Count accuracy per image
- Over/under-detection analysis
- Confidence score distributions

## Future Improvements

### Stage 2: QR Decoding & Classification (Not Implemented - Future Work)

- Implement QR code value extraction using pyzbar or similar
- Classify QR types (manufacturer, batch, distributor, regulator)
- Generate `submission_decoding_2.json` format
- Add string accuracy metrics for decoded values

### Model Enhancements

- Ensemble multiple models
- Custom anchor optimization
- Advanced augmentation strategies
- Multi-scale training

## Support & Contact

For questions about this implementation:

1. Check the evaluation script output for detailed metrics
2. Review visualization images for model performance
3. Check training logs for convergence issues

## One-Command Reproduction

For hackathon organizers to reproduce **Stage 1: Detection** results:

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Run inference on provided test images (Stage 1: Detection)
python infer.py --input path/to/test_images --output submission_detection_1.json --model outputs/best.pt

# 3. Validate submission format
python evaluate.py --predictions submission_detection_1.json --ground-truth data/combined.csv
```

**Expected Output**: `submission_detection_1.json` with detected QR code bounding boxes for all test images.

**Stage 1 JSON Format**:

```json
[
  {
    "image_id": "img001.jpg",
    "qrs": [
      {"bbox": [x_min, y_min, x_max, y_max]},
      {"bbox": [x_min, y_min, x_max, y_max]}
    ]
  }
]
```

### Tested & Verified

- **Model Location**: `outputs/best.pt` (99.16% F1 Score)
- **Inference Speed**: ~118ms per image
- **Format Validation**: Passes hackathon JSON format requirements
- **Demo Test**: Successfully detected 8 QR codes across 3 demo images
- **Stage**: Stage 1 (Detection) only - ready for submission
- **Stage 2**: Not implemented (bonus decoding task)
