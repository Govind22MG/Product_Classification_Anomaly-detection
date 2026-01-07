'''
    Product Classifier & Defect Classifier
    ...
      Date: 26-Sep-2024
    Author: SWAPPY404 <challaniswapnil98@gmail.com>

BSD 3-Clause License

Copyright (c) 2024, SWAPPY404

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Libraries
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as trans
import cnn_arch

# Image Transformer (Image Pixels --> Tensors)
img_transformer = trans.ToTensor()

# Model Path
DATA_FOLDER_PATH = 'data/'
MODEL_PATHS = {
    'product': f'{DATA_FOLDER_PATH}model/PC_MODEL_CEL.pt',
    'capsule': f'{DATA_FOLDER_PATH}model/CAPSULE_MODEL_CEL.pt',
    'leather': f'{DATA_FOLDER_PATH}model/LEATHER_MODEL_CEL.pt',
    'screw': f'{DATA_FOLDER_PATH}model/SCREW_MODEL_CEL.pt'
}

# Class-labels
products_labels = ['capsule', 'leather', 'screw']
condition_labels = ['defective', 'good']

# CNN Model Configs
MODEL_CONFIGS = {
    'product': {'in_channels': 1, 'out_channels': 3},
    'defect': {'in_channels': 1, 'out_channels': 2}
}

# Device configuration
DEVICE = 'cpu'

# Define model loading function
def _load_model(model_name, model_path, in_channels, out_channels, model_class):
    """Load a single model with error handling."""
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=DEVICE)
        model.eval()  # Set to evaluation mode
        print(f'{model_name} is Loaded ({model_path}).')
        return model
    else:
        print(f'ERROR: Unable to load {model_name} from {model_path}')
        raise FileNotFoundError(f'Model not found: {model_path}')

# CNN Instances & Loading
try:
    PC_NET = _load_model(
        'Product Classification Model',
        MODEL_PATHS['product'],
        MODEL_CONFIGS['product']['in_channels'],
        MODEL_CONFIGS['product']['out_channels'],
        cnn_arch.cnn_model
    )
    CAPSULE_NET = _load_model(
        'Capsule Defect Model',
        MODEL_PATHS['capsule'],
        MODEL_CONFIGS['defect']['in_channels'],
        MODEL_CONFIGS['defect']['out_channels'],
        cnn_arch.cnn_model_2
    )
    LEATHER_NET = _load_model(
        'Leather Defect Model',
        MODEL_PATHS['leather'],
        MODEL_CONFIGS['defect']['in_channels'],
        MODEL_CONFIGS['defect']['out_channels'],
        cnn_arch.cnn_model_2
    )
    SCREW_NET = _load_model(
        'Screw Defect Model',
        MODEL_PATHS['screw'],
        MODEL_CONFIGS['defect']['in_channels'],
        MODEL_CONFIGS['defect']['out_channels'],
        cnn_arch.cnn_model_2
    )
except FileNotFoundError as e:
    print(f'Fatal Error: {e}')
    exit(1)

# Image preprocessing and prediction helper function
def _predict_with_model(img, model, target_size=100):
    """
    Generic prediction function to reduce code duplication.
    
    Args:
        img: Input image (OpenCV format)
        model: PyTorch model for inference
        target_size: Target image size for resizing
    
    Returns:
        Tuple of (prediction probabilities, argmax index)
    """
    # Preprocess image
    img = cv2.resize(img, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to tensor
    with torch.no_grad():
        x = img_transformer(img).unsqueeze(0)
        y = model(x)
    
    # Convert to numpy and get predictions
    y = y.detach().cpu().numpy()[0]
    y_amax = np.argmax(y)
    
    return y, y_amax


# Product class prediction function
def product_class_predict(img):
    """Predict product classification."""
    y, y_amax = _predict_with_model(img, PC_NET)
    return y, y_amax, products_labels[y_amax]


# Capsule defect prediction function
def capsule_defect_predict(img):
    """Predict capsule defect classification."""
    y, y_amax = _predict_with_model(img, CAPSULE_NET)
    return y, y_amax, condition_labels[y_amax]


# Leather defect prediction function
def leather_defect_predict(img):
    """Predict leather defect classification."""
    y, y_amax = _predict_with_model(img, LEATHER_NET)
    return y, y_amax, condition_labels[y_amax]


# Screw defect prediction function
def screw_defect_predict(img):
    """Predict screw defect classification."""
    y, y_amax = _predict_with_model(img, SCREW_NET)
    return y, y_amax, condition_labels[y_amax]
