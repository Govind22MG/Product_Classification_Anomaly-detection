'''
    CNN Architecture for Classification
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
import numpy
import torch
import torch.nn as NN
import torch.nn.functional as  FN
from torchsummary import summary

class cnn_model(NN.Module):
    def __init__(self, in_channels, out_channels):
        super(cnn_model, self).__init__()
        
        # 100x100xin_channels --> 98x98x8
        self.conv1 = NN.Conv2d(in_channels, 8, kernel_size=(3, 3))
        # 98x98x8 --> 96x96x16 --> (MAXPOOL2D) --> 48x48x16
        self.conv2 = NN.Conv2d(8, 16, kernel_size=(3, 3))
        # 48x48x16 --> 46x46x32
        self.conv3 = NN.Conv2d(16, 32, kernel_size=(3, 3))
        # 46x46x32 --> 44x44x64 --> (MAXPOOL2D) --> 22x22x64
        self.conv4 = NN.Conv2d(32, 64, kernel_size=(3, 3))
        # 22x22x64 --> 20x20x128
        self.conv5 = NN.Conv2d(64, 128, kernel_size=(3, 3))
        # 20x20x128 --> 18x18x64 --> (MAXPOOL2D) --> 9x9x64
        self.conv6 = NN.Conv2d(128, 64, kernel_size=(3, 3))
        # 9x9x64 --> 7x7x32
        self.conv7 = NN.Conv2d(64, 32, kernel_size=(3, 3))
        # 7x7x32 --> 5x5x16 --> (MAXPOOL2D) --> 2x2x16
        self.conv8 = NN.Conv2d(32, 16, kernel_size=(3, 3))
        
        # Max-Pooling Layer
        self.maxpool = NN.MaxPool2d((2, 2))

        # Fully Connected Layers [ANN]
        self.lin1 = NN.Linear(2*2*16, 128)
        self.lin2 = NN.Linear(128, 32)
        self.lin3 = NN.Linear(32, out_channels)

        # Activation Functions
        self.softmax = NN.Softmax(dim=1)
        self.sigmoid = NN.Sigmoid()

    def forward(self, x):
        y = FN.relu(self.conv1(x)) # 
        y = FN.relu(self.conv2(y)) # 
        y = self.maxpool(y)        # 
        y = FN.relu(self.conv3(y)) # 
        y = FN.relu(self.conv4(y)) # 
        y = self.maxpool(y)        # 
        y = FN.relu(self.conv5(y)) # 
        y = FN.relu(self.conv6(y)) # 
        y = self.maxpool(y)        # 
        y = FN.relu(self.conv7(y)) # 
        y = FN.relu(self.conv8(y)) # 
        y = self.maxpool(y)        # 
        
        y = y.view(-1, 2*2*16)   # Flattening
        
        y = FN.relu(self.lin1(y))    # 
        y = FN.relu(self.lin2(y))    # 
        y = self.sigmoid(self.lin3(y)) # 
        y = self.softmax(y)     # Classification Probability Distributions
        return(y)

class cnn_model_2(NN.Module):
    def __init__(self, in_channels, out_channels):
        super(cnn_model_2, self).__init__()
        
        # 100x100xin_channels --> 98x98x16
        self.conv1 = NN.Conv2d(in_channels, 16, kernel_size=(3, 3))
        # 98x98x16 --> 96x96x64 --> (MAXPOOL2D) --> 48x48x64
        self.conv2 = NN.Conv2d(16, 64, kernel_size=(3, 3))
        # 48x48x64 --> 46x46x128
        self.conv3 = NN.Conv2d(64, 128, kernel_size=(3, 3))
        # 46x46x128 --> 44x44x256 --> (MAXPOOL2D) --> 22x22x256
        self.conv4 = NN.Conv2d(128, 256, kernel_size=(3, 3))
        # 22x22x256 --> 20x20x128
        self.conv5 = NN.Conv2d(256, 128, kernel_size=(3, 3))
        # 20x20x128 --> 18x18x64 --> (MAXPOOL2D) --> 9x9x64
        self.conv6 = NN.Conv2d(128, 64, kernel_size=(3, 3))
        # 9x9x64 --> 7x7x32
        self.conv7 = NN.Conv2d(64, 32, kernel_size=(3, 3))
        # 7x7x32 --> 5x5x16 --> (MAXPOOL2D) --> 2x2x16
        self.conv8 = NN.Conv2d(32, 16, kernel_size=(3, 3))
        
        # Max-Pooling Layer
        self.maxpool = NN.MaxPool2d((2, 2))

        # Fully Connected Layers [ANN]
        self.lin1 = NN.Linear(2*2*16, 256)
        self.lin2 = NN.Linear(256, 64)
        self.lin3 = NN.Linear(64, out_channels)

        # Activation Functions
        self.softmax = NN.Softmax(dim=1)
        self.sigmoid = NN.Sigmoid()

    def forward(self, x):
        y = FN.relu(self.conv1(x)) # 
        y = FN.relu(self.conv2(y)) # 
        y = self.maxpool(y)        # 
        y = FN.relu(self.conv3(y)) # 
        y = FN.relu(self.conv4(y)) # 
        y = self.maxpool(y)        # 
        y = FN.relu(self.conv5(y)) # 
        y = FN.relu(self.conv6(y)) # 
        y = self.maxpool(y)        # 
        y = FN.relu(self.conv7(y)) # 
        y = FN.relu(self.conv8(y)) # 
        y = self.maxpool(y)        # 
        
        y = y.view(-1, 2*2*16)   # Flattening
        
        y = FN.relu(self.lin1(y))    # 
        y = FN.relu(self.lin2(y))    # 
        y = self.sigmoid(self.lin3(y)) # 
        y = self.softmax(y)     # Classification Probability Distributions
        return(y)
