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



# Define the CNN-MHA-GRU classification architecture
class CustomGRU_Classifier_v1(nn.Module):
    def __init__(self, sample_length, total_axes, num_classes = 1,
                 d_gru1 = 48, d_gru2 = 48):
        super(CustomGRU_Classifier_v1, self).__init__()

        self.seq_len = sample_length
        self.in_channels = total_axes

        self.num_classes = num_classes

        dr1 = 0.2 # Dropout rate specified in architecture
        dr2 = 0.1 # intermediate Dropout rate adjusted as hyper parameter

        gru1_hd = d_gru1 # hidden length of GRU1
        gru2_hd = d_gru2 # hidden length of GRU2
        fcn_hd = 100
        
        self.cnn_block_1 = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(in_channels=total_axes,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.mp1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(p=dr1)
        
        self.cnn_block_2 = nn.Sequential(
            # Conv Block 2
            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.mp2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(p=dr1)

        self.cnn_block_3 = nn.Sequential(
            # Conv Block 3
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.drop3 = nn.Dropout(p=dr1)

        self.gru1 = nn.GRU(64,
                   gru1_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.ln1 = nn.LayerNorm(gru1_hd)
        self.drop4 = nn.Dropout(p=dr2)
    
        self.gru2 = nn.GRU(gru1_hd,
                   gru2_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.ln2 = nn.LayerNorm(gru2_hd)
        self.drop5 = nn.Dropout(p=dr1)

        self.fcn_output = nn.Sequential(
            nn.Flatten(), # Collapse Time singleton dimension from gru2
            nn.Linear(gru2_hd, fcn_hd),
            nn.ReLU(),
            # nn.Dropout(p=dr1),
            nn.Linear(fcn_hd, self.num_classes)
        )



    

    def forward(self, x):
        x = self.cnn_block_1(x) 
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.cnn_block_2(x) 
        x = self.mp2(x)
        x = self.drop2(x)
        
        x = self.cnn_block_3(x) 
        x = self.drop3(x)
        
        # Flip Dimensions to (batch, time, features) for Attention Layer
        x = x.transpose(1, 2)
        
        x, hidden_state_final = self.gru1(x)
        x = self.ln1(x)
        x = self.drop4(x)

        x, hidden_state_final = self.gru2(x)
        x = self.ln2(x)
        x = x[:, -1, :] # only grab final hidden state (return_sequences=FALSE)
        x = self.drop5(x)

        x = self.fcn_output(x)
        
        return x

# end of Class def GRU_CLassifier_v1


class CustomGRU_Classifier_v2(nn.Module):
    def __init__(self, sample_length, total_axes, num_classes = 1,
                  d_mha = 128, d_gru1 = 48, d_gru2 = 48):
        super(CustomGRU_Classifier_v2, self).__init__()

        self.seq_len = sample_length
        self.in_channels = total_axes

        self.num_classes = num_classes

        dr1 = 0.2 # Dropout rate specified in architecture
        dr2 = 0.1 # intermediate Dropout rate adjusted as hyper parameter

        gru1_hd = d_gru1 # hidden length of GRU1
        mha_d = d_mha # dimension of SelfAttention
        mha_numHeads = 4
        gru2_hd = d_gru2 # hidden length of GRU2
        fcn_hd = 100
        
        self.cnn_block_1 = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(in_channels=total_axes,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.mp1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(p=dr1)
        
        self.cnn_block_2 = nn.Sequential(
            # Conv Block 2
            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.mp2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(p=dr1)

        self.cnn_block_3 = nn.Sequential(
            # Conv Block 3
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.drop3 = nn.Dropout(p=dr1)



        self.attention_prep = nn.Linear(64, mha_d)
        self.attention_block = nn.MultiheadAttention(mha_d, mha_numHeads, batch_first = True)

        self.gru1 = nn.GRU(mha_d,
                   gru1_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.ln1 = nn.LayerNorm(gru1_hd)
        self.drop4 = nn.Dropout(p=dr2)
    
        

        self.gru2 = nn.GRU(gru1_hd,
                   gru2_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.ln2 = nn.LayerNorm(gru2_hd)
        self.drop5 = nn.Dropout(p=dr1)

        self.fcn_output = nn.Sequential(
            nn.Flatten(), # Collapse Time singleton dimension from gru2
            nn.Linear(gru2_hd, fcn_hd),
            nn.ReLU(),
            # nn.Dropout(p=dr1),
            nn.Linear(fcn_hd, self.num_classes)
        )



    

    def forward(self, x):
        x = self.cnn_block_1(x) 
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.cnn_block_2(x) 
        x = self.mp2(x)
        x = self.drop2(x)
        
        x = self.cnn_block_3(x) 
        x = self.drop3(x)
        
        # Flip Dimensions to (batch, time, features) for Attention Layer
        x = x.transpose(1, 2)

        x = self.attention_prep(x)
        # x, attn_output_weights
        x, _ = self.attention_block(x,x,x) # Pass all input as Q, K, V for self attention

        
        # x, hidden_state_final = self.gru1(x)
        x, _ = self.gru1(x)
        x = self.ln1(x)
        x = self.drop4(x)
        

        x, hidden_state_final = self.gru2(x)
        x = self.ln2(x)
        x = x[:, -1, :] # only grab final hidden state (return_sequences=FALSE)
        x = self.drop5(x)

        

        x = self.fcn_output(x)
        
        return x

# end of Class def GRU_CLassifier_v2



# Define the CNN-MHA-GRU classification architecture
class CustomGRU_Classifier_v3(nn.Module):
    def __init__(self, sample_length, total_axes, num_classes = 1,
                 d_gru1 = 48, d_gru2 = 48):
        super(CustomGRU_Classifier_v3, self).__init__()

        self.seq_len = sample_length
        self.in_channels = total_axes

        self.num_classes = num_classes

        dr1 = 0.2 # Dropout rate specified in architecture
        dr2 = 0.1 # intermediate Dropout rate adjusted as hyper parameter

        gru1_hd = d_gru1 # hidden length of GRU1
        gru2_hd = d_gru2 # hidden length of GRU2
        fcn_hd = 100
        
        self.cnn_block_1 = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(in_channels=total_axes,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.drop1 = nn.Dropout(p=dr1)
        
        self.cnn_block_2 = nn.Sequential(
            # Conv Block 2
            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.mp1 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(p=dr1)

        self.cnn_block_3 = nn.Sequential(
            # Conv Block 3
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.drop3 = nn.Dropout(p=dr1)

        self.gru1 = nn.GRU(64,
                   gru1_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.ln1 = nn.LayerNorm(gru1_hd)
        self.drop4 = nn.Dropout(p=dr2)
    
        self.gru2 = nn.GRU(gru1_hd,
                   gru2_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.ln2 = nn.LayerNorm(gru2_hd)
        self.drop5 = nn.Dropout(p=dr1)

        self.fcn_output = nn.Sequential(
            nn.Flatten(), # Collapse Time singleton dimension from gru2
            nn.Linear(gru2_hd, fcn_hd),
            nn.ReLU(),
            # nn.Dropout(p=dr1),
            nn.Linear(fcn_hd, self.num_classes)
        )



    

    def forward(self, x):
        x = self.cnn_block_1(x) 
        x = self.drop1(x)
        
        x = self.cnn_block_2(x) 
        x = self.mp1(x)
        x = self.drop2(x)
        
        x = self.cnn_block_3(x) 
        x = self.drop3(x)
        
        # Flip Dimensions to (batch, time, features) for Attention Layer
        x = x.transpose(1, 2)
        
        x, hidden_state_final = self.gru1(x)
        x = self.ln1(x)
        x = self.drop4(x)

        x, hidden_state_final = self.gru2(x)
        x = self.ln2(x)
        x = x[:, -1, :] # only grab final hidden state (return_sequences=FALSE)
        x = self.drop5(x)

        x = self.fcn_output(x)
        
        return x

# end of Class def GRU_CLassifier_v3

# Define the CNN-MHA-GRU classification architecture
class CustomGRU_Classifier_v4(nn.Module):
    def __init__(self, sample_length, total_axes, num_classes = 1,
                 d_gru1 = 48, d_gru2 = 48):
        super(CustomGRU_Classifier_v4, self).__init__()

        self.seq_len = sample_length
        self.in_channels = total_axes

        self.num_classes = num_classes

        dr1 = 0.2 # Dropout rate specified in architecture
        dr2 = 0.1 # intermediate Dropout rate adjusted as hyper parameter

        gru1_hd = d_gru1 # hidden length of GRU1
        gru2_hd = d_gru2 # hidden length of GRU2
        fcn_hd = 100
        
        self.cnn_block_1 = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(in_channels=total_axes,
                      out_channels=32,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.drop1 = nn.Dropout(p=dr1)
        
        self.cnn_block_2 = nn.Sequential(
            # Conv Block 2
            nn.Conv1d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.drop2 = nn.Dropout(p=dr1)

        self.cnn_block_3 = nn.Sequential(
            # Conv Block 3
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU()
        )
        self.drop3 = nn.Dropout(p=dr1)

        self.gru1 = nn.GRU(64,
                   gru1_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.ln1 = nn.LayerNorm(gru1_hd)
        self.drop4 = nn.Dropout(p=dr2)
    
        self.gru2 = nn.GRU(gru1_hd,
                   gru2_hd,
                   batch_first=True,
                   # dropout=0.5,
                   bidirectional = False)
        self.ln2 = nn.LayerNorm(gru2_hd)
        self.drop5 = nn.Dropout(p=dr1)

        self.fcn_output = nn.Sequential(
            nn.Flatten(), # Collapse Time singleton dimension from gru2
            nn.Linear(gru2_hd, fcn_hd),
            nn.ReLU(),
            # nn.Dropout(p=dr1),
            nn.Linear(fcn_hd, self.num_classes)
        )



    

    def forward(self, x):
        x = self.cnn_block_1(x) 
        x = self.drop1(x)
        
        x = self.cnn_block_2(x)
        x = self.drop2(x)
        
        x = self.cnn_block_3(x) 
        x = self.drop3(x)
        
        # Flip Dimensions to (batch, time, features) for Attention Layer
        x = x.transpose(1, 2)
        
        x, hidden_state_final = self.gru1(x)
        x = self.ln1(x)
        x = self.drop4(x)

        x, hidden_state_final = self.gru2(x)
        x = self.ln2(x)
        x = x[:, -1, :] # only grab final hidden state (return_sequences=FALSE)
        x = self.drop5(x)

        x = self.fcn_output(x)
        
        return x

# end of Class def GRU_CLassifier_v4
    

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

def Create_CustomGRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 0):

    # Output Classes
    num_outputs = 1 # Single value intake or not
    # num_outputs = 2 # predict bites and drinks separately



    if MODEL_ARCH_FLAG == 0:
        # Add 3 CNN Blocks
        model = CustomGRU_Classifier_v1(sample_length, total_axes, num_classes = num_outputs)
    elif MODEL_ARCH_FLAG == 1:
        # Add 3 CNN Blocks, add attention with 128 dim
        model = CustomGRU_Classifier_v2(sample_length, total_axes, num_classes = num_outputs)
    elif MODEL_ARCH_FLAG == 3:
        # Add 3 CNN Blocks, add attention with 128 dim
        model = CustomGRU_Classifier_v3(sample_length, total_axes, num_classes = num_outputs)
    elif MODEL_ARCH_FLAG == 4:
        # Add 3 CNN Blocks, add attention with 128 dim
        model = CustomGRU_Classifier_v4(sample_length, total_axes, num_classes = num_outputs)
    # end of switch MODEL_ARCH_FLAG
    
    return model

# end of def Create_CustomGRU_Classifier

    
