# Product Classification & Anomaly Detection (Defect Identification)

## Overview
This project implements a two-stage deep learning system for **product classification** and **defect detection** using Convolutional Neural Networks (CNNs). It classifies products into categories (capsule, leather, screw) and identifies defects/anomalies within each product category using pre-trained models.

The application features a **Tkinter-based GUI** for easy interaction and visualization of predictions with confidence scores.

## Dataset
- **Dataset Used**: MVTEC Anomaly Detection (MVTEC-AD)
- **License**: CC BY-NC-SA 4.0 (Non-commercial use only)
- **Products Covered**: Capsule, Leather, Screw
- **Defect Types**:
  - **Capsule**: crack, faulty_imprint, poke, scratch, squeeze
  - **Leather**: color, cut, fold, glue, poke
  - **Screw**: manipulated_front, scratch_head, scratch_neck, thread_side, thread_top

## Project Structure

```
product_classification_anomaly_detection/
├── main.py                          # GUI application entry point
├── product_defect_classifier.py     # Classification logic & model inference
├── cnn_arch.py                      # CNN architecture definitions
├── cnn_arch_summary.py              # Model summary utility
├── data/
│   ├── model/                       # Pre-trained model weights
│   │   ├── PC_MODEL_CEL.pt         # Product Classification Model
│   │   ├── CAPSULE_MODEL_CEL.pt    # Capsule Defect Detection Model
│   │   ├── LEATHER_MODEL_CEL.pt    # Leather Defect Detection Model
│   │   └── SCREW_MODEL_CEL.pt      # Screw Defect Detection Model
│   └── mvtec_ad_test_dataset/       # Test dataset
```

## File Descriptions

### 1. **main.py** - GUI Application
The main entry point that provides a user-friendly Tkinter interface for:
- **Image/Directory Selection**: Browse and select single images or directories
- **Real-time Predictions**: Display product classification and defect detection
- **Confidence Scores**: Show prediction probabilities for each class
- **Multi-image Navigation**: View predictions for multiple images with a configurable interval
- **Key Features**:
  - DISPLAY_IMAGE_INTERVAL = 2.0 seconds between images
  - Displays product classification scores for all classes
  - Highlights the predicted class in red
  - Shows defect classification when a product type is identified

### 2. **product_defect_classifier.py** - Core Classification Engine
Contains all model initialization and prediction functions:

#### Models Loaded:
- `PC_NET`: Product Classification Network (1 input channel → 3 output classes)
- `CAPSULE_NET`: Capsule Defect Classifier (1 input channel → 2 output classes)
- `LEATHER_NET`: Leather Defect Classifier (1 input channel → 2 output classes)
- `SCREW_NET`: Screw Defect Classifier (1 input channel → 2 output classes)

#### Key Functions:
- `product_class_predict(img)` - Classifies product type (capsule/leather/screw)
- `capsule_defect_predict(img)` - Detects defects in capsules
- `leather_defect_predict(img)` - Detects defects in leather
- `screw_defect_predict(img)` - Detects defects in screws

#### Processing Pipeline:
1. Resize image to 100×100 pixels
2. Convert RGB to Grayscale
3. Convert to tensor format
4. Forward pass through CNN
5. Convert output to NumPy array
6. Apply argmax to get predicted class
7. Return predictions with confidence scores

### 3. **cnn_arch.py** - Neural Network Architectures
Defines two CNN architectures for classification tasks:

#### **cnn_model** (Product Classification Network)
- **Input**: 100×100×1 (grayscale image)
- **Output**: 3 classes (capsule, leather, screw)
- **Architecture**:
  - 8 convolutional layers with ReLU activation
  - 4 max-pooling layers for dimension reduction
  - 3 fully connected layers
  - Sigmoid activation followed by Softmax for probability distribution
  - Progressive channel expansion: 1→8→16→32→64→128→64→32→16
  - Final flattened size: 2×2×16 = 64 features

#### **cnn_model_2** (Defect Detection Network)
- **Input**: 100×100×1 (grayscale image)
- **Output**: 2 classes (good/defective)
- **Architecture**:
  - 8 convolutional layers with ReLU activation
  - 4 max-pooling layers
  - 3 fully connected layers
  - Sigmoid activation followed by Softmax
  - Channel expansion: 1→16→64→128→256→128→64→32→16
  - Final flattened size: 2×2×16 = 64 features

Both models use similar design patterns but differ in channel depths and output classes.

### 4. **cnn_arch_summary.py**
Utility script to display model architecture summaries and layer information.

## How It Works

### Two-Stage Pipeline:

```
Input Image
    ↓
[Stage 1: Product Classification]
    - Classify: Capsule / Leather / Screw
    ↓
[Stage 2: Defect Detection]
    - Based on product type, use appropriate model
    - Classify: Good / Defective
    ↓
Output: Product Type + Defect Status + Confidence Scores
```

## Requirements

```
opencv-python (cv2)
torch (PyTorch)
torchvision
numpy
pillow (PIL)
tkinter (usually included with Python)
torchsummary (optional)
```

## Usage

### Running the Application:
```bash
python main.py
```

### GUI Controls:
1. **Browse Image**: Select a single image file for prediction
2. **Browse Folder**: Select a directory to process all images
3. **Navigation Buttons**: Move through multiple images
4. **Exit**: Close the application

### Predictions Output:
- **Product Classification**: Shows confidence scores for each product type
- **Defect Classification**: Shows whether product is good or defective with confidence
- **Real-time Display**: Updates with each image selection/navigation

## Model Information

### Input/Output Specifications:
- **Input Size**: All models expect 100×100 grayscale images
- **Input Format**: OpenCV images (BGR) are converted to grayscale
- **Output Format**: Probability scores (0-1) for each class

### Performance:
- **Inference Speed**: Real-time predictions on CPU
- **Memory**: Minimal memory footprint for embedded/edge deployment
- **Accuracy**: Trained on MVTEC-AD dataset for industrial anomaly detection

## Key Technical Features

1. **Image Preprocessing**:
   - Automatic resizing to 100×100
   - RGB to Grayscale conversion
   - Tensor normalization (PyTorch)

2. **Neural Network Layers**:
   - Convolutional layers for feature extraction
   - ReLU activation functions for non-linearity
   - Max-pooling for spatial dimension reduction
   - Fully connected layers for classification
   - Softmax output for probability distribution

3. **GUI Features**:
   - Real-time image display
   - Dynamic confidence score visualization
   - Multi-image batch processing
   - Configurable display interval

## License
BSD 3-Clause License - See header in source files for details

**Note**: Dataset is CC BY-NC-SA 4.0 (Non-commercial use only)

## Author
SWAPPY404 <challaniswapnil98@gmail.com>
Date: September 30, 2024

---

## Additional Notes

- Models must be pre-trained and stored in `data/model/` directory
- Application expects grayscale images or will convert color images automatically
- For best results, use images similar to MVTEC-AD dataset (industrial products)
- GUI is optimized for 960×480 resolution
- Processing interval can be adjusted via `DISPLAY_IMAGE_INTERVAL` variable
