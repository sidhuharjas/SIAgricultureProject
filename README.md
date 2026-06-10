# SI Agriculture Project

A comprehensive machine learning pipeline for automated crop and weed detection in agricultural imagery using YOLO object detection.

## 🌾 Overview

This project implements an end-to-end solution for classifying crops and weeds in agricultural images. It combines computer vision techniques with YOLO (You Only Look Once) deep learning architecture to enable efficient, real-time detection of vegetation types in farm environments.

### Key Features

- **Automated Image Annotation**: Uses Excess Green Index (ExG) for vegetation detection
- **Data Pipeline**: Complete workflow from raw images to trained models
- **YOLO Integration**: YOLOv8 compatible dataset and training setup
- **Image Preprocessing**: Advanced preprocessing techniques for improved detection
- **Data Cleaning**: Tools for cleaning and validating annotated data

## 📁 Project Structure

```
SIAgricultureProject/
├── auto_Anotate.py           # Automated YOLO annotation using ExG index
├── cleanUp.py                 # Data cleaning and validation utilities
├── image_preprocessing.py      # Image enhancement and preprocessing
├── labeler.py                 # Interactive labeling tools
├── tunedLabler.py             # Fine-tuned labeling with advanced features
├── prep_pipeline.py            # Data preparation pipeline
├── train_pipeline.py           # Training orchestration
├── train_yolo.py              # YOLO model training script
├── getData.py                 # Data acquisition utilities
├── getData2.py                # Alternative data fetching
├── export_lightweight.py       # Export models for deployment
├── data.yaml                  # Dataset configuration
├── dataset.yaml               # Alternative dataset config
├── crop_weed_data/            # Symlink to crop/weed dataset
├── weeds_data/                # Symlink to weed-specific dataset
└── Comands to run             # Quick reference commands
```

## 🛠️ Core Scripts

### `auto_Anotate.py`
Automatically generates YOLO-format annotations for images using computer vision.

**Features:**
- Excess Green Index (ExG) for vegetation detection
- Otsu thresholding for robust segmentation
- Morphological operations to remove noise
- Size-based classification (crops vs. weeds)
- Generates YOLO `.txt` label files

**Usage:**
```bash
python auto_Anotate.py
```

### `image_preprocessing.py`
Comprehensive image preprocessing for improved model performance.

### `prep_pipeline.py`
Orchestrates the complete data preparation workflow.

### `train_pipeline.py` & `train_yolo.py`
Handles model training with YOLO architecture.

**Usage:**
```bash
python train_yolo.py
```

### `cleanUp.py`
Data validation and cleanup utilities.

### `labeler.py` & `tunedLabler.py`
Interactive tools for manual annotation and label refinement.

## 🎯 Classification

The system classifies vegetation into two categories:

- **Class 0 (CROP)**: Larger plant organisms (area > 4500 pixels)
- **Class 1 (WEED)**: Smaller unwanted vegetation

## 📊 Dataset Configuration

Two YAML configurations are available:

- `data.yaml`: Primary dataset configuration
- `dataset.yaml`: Alternative dataset setup

Both are compatible with YOLO training pipelines.

## 🚀 Quick Start

### 1. Automatic Annotation
Annotate a folder of images automatically:
```bash
python auto_Anotate.py
```

### 2. Data Preparation
Prepare and validate your dataset:
```bash
python prep_pipeline.py
```

### 3. Train Model
Train a YOLO model on your dataset:
```bash
python train_yolo.py
```

### 4. Export Lightweight Model
Export an optimized model for deployment:
```bash
python export_lightweight.py
```

## 📋 Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- YOLO library (YOLOv8)
- Additional dependencies as specified in training scripts

## 🔧 Configuration

Edit the configuration sections in each script to customize:

- Dataset paths
- Classification thresholds
- Image preprocessing parameters
- Training hyperparameters

Key configuration in `auto_Anotate.py`:
```python
DATASET_DIR = "/path/to/images"
CROP_CLASS_ID = 0
WEED_CLASS_ID = 1
```

## 📝 Output Format

The pipeline generates YOLO-format annotations:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where all coordinates are normalized to the image dimensions (0-1 range).

## 🔄 Workflow

1. **Data Collection** → Gather agricultural images
2. **Auto-Annotation** → Generate initial labels with `auto_Anotate.py`
3. **Manual Review** → Refine labels with `labeler.py`
4. **Preprocessing** → Enhance images with `image_preprocessing.py`
5. **Preparation** → Organize data with `prep_pipeline.py`
6. **Training** → Train YOLO model with `train_yolo.py`
7. **Export** → Optimize model with `export_lightweight.py`

## 📦 Technologies

- **YOLOv8**: Real-time object detection
- **OpenCV**: Computer vision processing
- **Python**: Core programming language

## 📄 License

Not specified. See repository for license details.

## 👤 Author

Created by **sidhuharjas**

## 📞 Support

For issues, questions, or contributions, please open an issue in the repository.

---

**Note**: Ensure all dataset paths are correctly configured before running scripts. Refer to `Comands to run` file for additional execution instructions.
