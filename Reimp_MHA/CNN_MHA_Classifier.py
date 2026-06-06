'''
Filename: CNN_MHA_Classifier.py
Author: James Jolly
Purpose: Implement PyTorch model of the Human Activity Recognition from IMU data found in the paper,
        "Convolutional Nerual Network With Multihead Attention for Human Activity Recognition" by Tan-Hsu Tan et al.
Version: 
    0.1: Intial Code
'''


import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR



# Define the CNN-MHA classification architecture
# The `feature_extractor` argument takes images as input and produces feature vectors.
# These vectors can be of any length. 
class CNN_MHA_Classifier(nn.Module):
    def __init__(self, feature_extractor, attention_prep, attention_block, regression_block):
        super(CNN_MHA_Classifier, self).__init__()

        # feature encoder
        self.feature_extractor = feature_extractor

        # Reshapes data to match the dimensions of attention block
        self.attention_prep = attention_prep

        # Multi Head Attention Block 
        self.attention_block = attention_block

        # Regression to final output
        self.regression_block = regression_block


    def forward(self, x):
        x = self.feature_extractor(x) 
        
        # Flip Dimensions to (batch, time, features) for Attention Layer
        x = x.transpose(1, 2)
        x = self.attention_prep(x)
        x, attn_output_weights = self.attention_block(x,x,x) # Pass all input as Q, K, V for self attention
        
        x = self.regression_block(x)
        return x

# end of Class def CNN_MHA_CLassifier



'''
Func: Create_CNN_MHA()
Purpose: Generates Pytorch Model with CNN-MHA for activity recognition
    INPUTS
sample_length - [int]
    Number of samples in a window
total_axes - [int]
    Number of features of the input time series
        
    OUTPUT
model  - PyTorch model using CNN-MHA for Human Actvitiy Recognition
'''

def Create_CNN_MHA(sample_length, total_axes):
    
    dropoutRate = 0.5

    # Output Classes
    num_outputs = 1 # Single value intake or not
    # num_outputs = 2 # predict bites and drinks separately
    

    CNN_Block = nn.Sequential(
        # Conv Block 1
        nn.Conv1d(in_channels=total_axes,
                  out_channels=64,
                  kernel_size=3,
                  stride=1),
        nn.LazyBatchNorm1d(),
        nn.ReLU(),
        # nn.Dropout(p=dropoutRate),

        nn.MaxPool1d(3),

        # Conv Block 2
        nn.Conv1d(in_channels=64,
                  out_channels=128,
                  kernel_size=3,
                  stride=1),
        nn.LazyBatchNorm1d(),
        nn.ReLU(),
        # nn.Dropout(p=dropoutRate),

        nn.MaxPool1d(3),

        # Conv Block 3
        nn.Conv1d(in_channels=128,
                  out_channels=256,
                  kernel_size=3,
                  stride=1),
        nn.LazyBatchNorm1d(),
        nn.ReLU(),
        # nn.Dropout(p=dropoutRate),

        nn.MaxPool1d(3)

    )

    d_model = 128
    num_heads = 16
    num_dense_hidden = 128

    MHA_Prep = nn.Linear(256, d_model)

    MHA_Block = nn.MultiheadAttention(d_model, num_heads, batch_first = True)
    
    Dense_Block = nn.Sequential(
        nn.Dropout(p=dropoutRate),
        nn.Flatten(),
        nn.LazyLinear(num_dense_hidden),
        nn.ReLU()
        nn.Linear(num_dense_hidden, num_outputs)
    )


    # Put together all blocks into model
    model =  CNN_MHA_Classifier(CNN_Block, MHA_Prep, MHA_Block, Dense_Block)

    return model

# end of def Create_CNN_MHA

    