'''
Filename: GRU_Classifier.py
Author: James Jolly
Purpose: Implement PyTorch model of the Human Activity Recognition from IMU data found in the paper,
        "Explainable CNN-GRU Model With Self-Attention for Human Activity Recognition" by Nadia Sultana et al.
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
class GRU_Classifier(nn.Module):
    def __init__(self, sample_length, total_axes, num_classes = 1, FINAL_ACTIVATION_FLAG = True,
                 d_gru1 = 48, d_mha = 96, d_gru2 = 48,
                droprate1 = 0.5): # Dropout rate specified in architecture
        super(GRU_Classifier, self).__init__()

        self.seq_len = sample_length
        self.in_channels = total_axes

        self.num_classes = num_classes
        self.FINAL_ACTIVATION_FLAG = FINAL_ACTIVATION_FLAG 

        dr1 = droprate1 # Dropout rate specified in architecture
        dr2 = 0.1 # intermediate Dropout rate adjusted as hyper parameter

        gru1_hd = d_gru1 # hidden length of GRU1
        mha_d = d_mha # dimension of SelfAttention
        mha_numHeads = 4
        gru2_hd = d_gru2 # hidden length of GRU2
        fcn_hd = 100
        


        self.cnn_block = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(in_channels=total_axes,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.drop1 = nn.Dropout(p=dr1)

        self.gru1 = nn.GRU(32,
                   gru1_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.drop2 = nn.Dropout(p=dr2)
        

    
        self.attention_prep = nn.Linear(gru1_hd, mha_d)

        self.attention_block = nn.MultiheadAttention(mha_d, mha_numHeads, batch_first = True)


        self.gru2 = nn.GRU(mha_d,
                   gru2_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.drop3 = nn.Dropout(p=dr1)
        

        

        self.fcn_output = nn.Sequential(
            nn.Flatten(), # Collapse Time singleton dimension from gru2
            nn.Linear(gru2_hd, fcn_hd),
            nn.ReLU(),
            # nn.Dropout(p=dr1),
            nn.Linear(fcn_hd, self.num_classes)
        )


        if self.num_classes == 1:
            self.final_activation = nn.Sigmoid()
        else: # if self.num_classes > 1
            self.final_activation = nn.Softmax()
        # end of if num_classes


    

    def forward(self, x):
        x = self.cnn_block(x) 
        x = self.drop1(x)

        # Flip Dimensions to (batch, time, features) for Attention Layer
        x = x.transpose(1, 2)
        
        x, hidden_state_final = self.gru1(x)
        x = self.drop2(x)

        x = self.attention_prep(x)
        x, attn_output_weights = self.attention_block(x,x,x) # Pass all input as Q, K, V for self attention

        x, hidden_state_final = self.gru2(x)
        x = x[:, -1, :] # only grab final hidden state (return_sequences=FALSE)
        x = self.drop3(x)

        x = self.fcn_output(x)
        
        # extra conditional for TallyNet to allow larger final outputs
        if (self.FINAL_ACTIVATION_FLAG == True):  
            x = self.final_activation(x)
        # end if
        
        return x

# end of Class def GRU_CLassifier



'''
Func: Create_GRU_Classifier()
Purpose: Generates Pytorch Model with CNN-MHA for activity recognition
    INPUTS
sample_length - [int]
    Number of samples in a window
total_axes - [int]
    Number of features of the input time series
        
    OUTPUT
model  - PyTorch model using CNN-MHA for Human Actvitiy Recognition
'''

def Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 0):

    # Output Classes
    num_outputs = 1 # Single value intake or not
    # num_outputs = 2 # predict bites and drinks separately
    
    
    if MODEL_ARCH_FLAG == 0: # Default
        # Close to Upper param count in original paper
        model =  GRU_Classifier(sample_length, total_axes, num_classes = num_outputs)
    elif MODEL_ARCH_FLAG == 1:
        # Smaller size used close to lower param count in original paper
        model =  GRU_Classifier(sample_length, total_axes, num_classes = 1, 
                FINAL_ACTIVATION_FLAG = True,
                d_gru1 = 32, d_mha = 64, d_gru2 = 32)
    elif MODEL_ARCH_FLAG == 2:
        model =  GRU_Classifier(sample_length, total_axes, num_classes = 1, 
                FINAL_ACTIVATION_FLAG = True,
                d_gru1 = 128, d_mha = 256, d_gru2 = 128)
    elif MODEL_ARCH_FLAG == 3:
        model =  GRU_Classifier(sample_length, total_axes, num_classes = num_outputs, droprate1 = 0.2)
    # end of switch MODEL_ARCH_FLAG
    
    return model

# end of def Create_GRU_Classifier

    
