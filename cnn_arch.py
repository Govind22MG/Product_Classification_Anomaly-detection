'''
    CNN Architecture for Classification
    ...
      Date: 26-Sep-2024
    Author: SWAPPY404 <challaniswapnil98@gmail.com>

BSD 3-Clause License
...
'''

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torchsummary import summary


class BaseCNNModel(nn.Module):
    """Base CNN class to reduce code duplication."""
    
    def __init__(self, in_channels, out_channels, conv_dims, fc_dims):
        super(BaseCNNModel, self).__init__()
        
        # Build convolutional layers dynamically
        self.conv_layers = nn.ModuleList()
        prev_channels = in_channels
        
        for out_ch in conv_dims:
            self.conv_layers.append(nn.Conv2d(prev_channels, out_ch, kernel_size=3))
            prev_channels = out_ch
        
        # Max-pooling layer
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        prev_size = 64  # Flattened size: 2*2*16
        
        for hidden_size in fc_dims:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        self.output = nn.Linear(prev_size, out_channels)
        
        # Activation functions
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional layers with pooling
        for i, conv_layer in enumerate(self.conv_layers):
            x = fn.relu(conv_layer(x))
            # Apply pooling after specific layers for dimension reduction
            if (i + 1) % 2 == 0:
                x = self.maxpool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = fn.relu(fc_layer(x))
        
        # Output
        x = self.sigmoid(self.output(x))
        x = self.softmax(x)
        
        return x


class cnn_model(BaseCNNModel):
    """Product Classification Network (3-class classifier)."""
    
    def __init__(self, in_channels=1, out_channels=3):
        # Channel dimensions: 1→8→16→32→64→128→64→32→16
        conv_dims = [8, 16, 32, 64, 128, 64, 32, 16]
        # FC layers: 64→128→32→3
        fc_dims = [128, 32]
        super(cnn_model, self).__init__(in_channels, out_channels, conv_dims, fc_dims)


class cnn_model_2(BaseCNNModel):
    """Defect Detection Network (binary classifier)."""
    
    def __init__(self, in_channels=1, out_channels=2):
        # Channel dimensions: 1→16→64→128→256→128→64→32→16
        conv_dims = [16, 64, 128, 256, 128, 64, 32, 16]
        # FC layers: 64→256→64→2
        fc_dims = [256, 64]
        super(cnn_model_2, self).__init__(in_channels, out_channels, conv_dims, fc_dims)
