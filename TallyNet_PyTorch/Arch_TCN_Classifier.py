'''
Filename: TCCSNet_Classifier.py
Author: James Jolly
Purpose: Implement PyTorch model of the Human Activity Recognition from IMU data found in the paper,
        "Temporal-channel convolution with self-attention network for juman activity recognition using wearable sensors" by Ehab Esaa and Islam R. Abdelmaksoud
Version: 
    0.2: Updated to Seq2One output, only using the middle output (with full convolution receptive field in view) as the output  
    0.1: Intial Code
'''

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR





class TemporalConvBlock(nn.Module):
    """
    CNN block used throughout TCCSnet architecture

    Expects inputs to be of order [Batch_size, Features, Seq_Length]
    """

    def __init__(self, d_tcn, layerDepth, droprate = 0.3):
        super().__init__()

        if layerDepth >= 1:
            levelDilation = 2 ** (layerDepth -1)
        else:
            levelDilation = 0 # if invalid layer depth, use no dilation

        
        self.conv1 = nn.Conv1d(d_tcn, d_tcn, 3, padding = 'same', dilation = levelDilation) # Dilated Kernel
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=droprate)
        self.conv2 = nn.Conv1d(d_tcn, d_tcn, 1) # 1x1 kernel
        self.bn = nn.BatchNorm1d(d_tcn)
        

    def forward(self, x):

        res = x.clone()
        
        x = self.conv1(x) # Dilated Conv
        x = self.drop(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        x = x + res

        x = self.bn(x) # normalize for stability

        
        return x
# end of def class TemporalConvBlock()


class TCN_BlockLayer(nn.Module):
    """
    One layer consisting of multiple Temporal Convolution Blocks
        INPUT
    num_blocks - [int] number of TCN Blocks to stack 
    d_tcn - [int] number of channels throughout the TCN Blocks
    """

    def __init__(self, num_blocks, d_tcn):
        super().__init__()

        blocks = []
        blocks.append(
            TemporalConvBlock(
                d_tcn, 1
            )
        )

        for currLayer in range(2,num_blocks+1):
            blocks.append(
                TemporalConvBlock(
                    d_tcn, currLayer
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
# end of def class TemporalConvBlock


        
#
class SelfAttentionBlock(nn.Module):
    """
    Self Attention Block used in Eating Speed TCN architecture

    Expects inputs to be of order [Batch_size, Seq_Length, Features]
    """

    def __init__(self, d_head, num_heads, drop_rate):
        super().__init__()

        # standard in paper
        # d_k = 16
        # num_heads = 8
        # d_model = 128

        self.d_head = d_head # dq = dk = dimension of Query and Keys
        self.num_heads = num_heads
        self.d_model = d_head * num_heads
        self.d_ff = 32 # number of hidden features in Feed Forward Network
    
    
        self.MHA_Block = nn.MultiheadAttention(self.d_model, self.num_heads, batch_first = True)


        self.Dropout = nn.Dropout(p=drop_rate)

        self.LayerNorm = nn.LayerNorm([self.d_model])
        
    def forward(self, x):

        # Pass data through MHA with residual
        
        res = x.clone() # Save residual 
        x, _ = self.MHA_Block(x, x, x) # Pass all input as Q, K, V for self attention  
        x = self.Dropout(x)

        # Add and Normalize
        x = x + res # Add residual back
        x = self.LayerNorm(x)
        
        return x
# end of def class SelfAttentionBlock()




class AddPosEncodingBlock(nn.Module):
    """
    Adds Positional Encoding to data. Origninally written for TCCSnet architecture, but added to TCN model

    Expects inputs to be of order [Batch_size, Features, Seq_Length]

    Modified Example from @hunter-j-phillips on Medium
    """

    def __init__(self, d_model: int, max_length: int = 5000):
        """
        Args:
          d_model:      dimension of embeddings
          dropout:      randomly zeroes-out some of the input
          max_length:   max sequence length
        """
        # inherit from Module
        super().__init__()     
        
        # create tensor of 0s
        pe = torch.zeros(max_length, d_model)    
        
        # create position column   
        k = torch.arange(0, max_length).unsqueeze(1)  
        # calc divisor for positional encoding 
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        # calc sine on even indices
        pe[:, 0::2] = torch.sin(k * div_term)    
        # calc cosine on odd indices   
        pe[:, 1::2] = torch.cos(k * div_term)  
        
        # add dimension     
        pe = pe.unsqueeze(0)          
        
        # buffers are saved in state_dict but not trained by the optimizer                        
        self.register_buffer("pe", pe)                        
    
    def forward(self, x):
        """
        Args:
          x:        embeddings (batch_size, seq_length, d_model)
        
        Returns:
                    embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        # add positional encoding to the embeddings
        x = x + self.pe[:, : x.size(1)].requires_grad_(False) 
        
        # perform dropout
        return x
# end of def class AddPosEncodingBlock()




# Define the TCCSnet classification architecture
class TCN_Classifier(nn.Module):
    def __init__(self, sample_length, total_axes, num_tcn_layers, FINAL_ACTIVATION_FLAG):
        super(TCN_Classifier, self).__init__()

        self.FINAL_ACTIVATION_FLAG = FINAL_ACTIVATION_FLAG
        self.num_classes = 1 # final output is only prediciting intake or not

        self.drop_rate = 0.3 

        self.d_tcn = 64 # number of filters to use for TCN blocks

        self.d_head = 16
        self.num_heads = 8
        self.d_model = self.d_head * self.num_heads
        self.d_ff_final = 64 # Num hidden neurons in final FFN block

        self.in_channels = total_axes
        self.in_seq_len = sample_length

        # Blocks
        self.TCN_Prep = nn.Conv1d(self.in_channels, self.d_tcn, 1) # Project into d_TCN Dimensions

        self.TCN_Layers = TCN_BlockLayer(num_tcn_layers, self.d_tcn)

        # Project to d_model
        self.MHA_Prep = nn.Linear(self.d_tcn, self.d_model) # Time

        self.PosEncode = AddPosEncodingBlock(self.d_model, sample_length)

        self.SelfAtten = SelfAttentionBlock(self.d_head, self.num_heads, self.drop_rate) # Time

        self.LayerNorm = nn.LayerNorm([self.d_model])
    
        # Final Dense Layer of shape [BatchSize, Neurons]
        self.FFN = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff_final),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.d_ff_final, self.num_classes)
        )

        if self.num_classes == 1:
            self.FinalActivation = nn.Sigmoid()
        else: # if self.num_classes > 1
            self.FinalActivation = nn.Softmax()
        # end of if num_classes


        
    def forward(self, x):

        x = self.TCN_Prep(x) 
        x = self.TCN_Layers(x)

         # Manage dimensions to match PosEncoder and MHA expected input 
        # [Batch_size, Features, Seq_Length] -> [Batch_size, Seq_Length, Features]
        x = x.transpose(1, 2)

        
        x = self.MHA_Prep(x) # Project to d_model

        x = self.LayerNorm(x) # normalize before encoding
        
        x = self.PosEncode(x)
        x = self.SelfAtten(x) # Pass all input as Q, K, V for self attention

        x = self.FFN(x)

        # extra conditional for TallyNet to allow larger final outputs
        if (self.FINAL_ACTIVATION_FLAG == True):  
            x = self.FinalActivation(x)
        # end if
        
        return x

# end of Class def CNN_MHA_CLassifier



# Define the TCCSnet classification architecture, output to a single time point in the middle of the input sequence
class TCN_Seq2OneClassifier(nn.Module):
    def __init__(self, sample_length, total_axes, num_tcn_layers, FINAL_ACTIVATION_FLAG):
        super(TCN_Seq2OneClassifier, self).__init__()

        self.output_idx = int((sample_length-1)/2) # midpoint of sequence
        self.FINAL_ACTIVATION_FLAG = FINAL_ACTIVATION_FLAG

        self.num_classes = 1 # final output is only prediciting intake or not

        self.drop_rate = 0.3 

        self.d_tcn = 64 # number of filters to use for TCN blocks

        self.d_head = 16
        self.num_heads = 8
        self.d_model = self.d_head * self.num_heads
        self.d_ff_final = 64 # Num hidden neurons in final FFN block

        self.in_channels = total_axes
        self.in_seq_len = sample_length

        # Blocks
        self.TCN_Prep = nn.Conv1d(self.in_channels, self.d_tcn, 1) # Project into d_TCN Dimensions

        self.TCN_Layers = TCN_BlockLayer(num_tcn_layers, self.d_tcn)

        # Project to d_model
        self.MHA_Prep = nn.Linear(self.d_tcn, self.d_model) # Time

        self.PosEncode = AddPosEncodingBlock(self.d_model, sample_length)

        self.SelfAtten = SelfAttentionBlock(self.d_head, self.num_heads, self.drop_rate) # Time

        self.LayerNorm = nn.LayerNorm([self.d_model])
    
        # Final Dense Layer of shape [BatchSize, Neurons]
        self.FFN = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff_final),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(self.d_ff_final, self.num_classes)
        )

        if self.num_classes == 1:
            self.FinalActivation = nn.Sigmoid()
        else: # if self.num_classes > 1
            self.FinalActivation = nn.Softmax()
        # end of if num_classes


        
    def forward(self, x):

        x = self.TCN_Prep(x) 
        x = self.TCN_Layers(x)

        # Remove sequence length to single timestep (midpoint in seq2seq)
        x = x.narrow(-1, self.output_idx, 1) 

         # Manage dimensions to match PosEncoder and MHA expected input 
        # [Batch_size, Features, Seq_Length] -> [Batch_size, Seq_Length, Features]
        x = x.transpose(1, 2)

        
        x = self.MHA_Prep(x) # Project to d_model

        x = self.LayerNorm(x) # normalize before encoding
        
        x = self.PosEncode(x)
        x = self.SelfAtten(x) # Pass all input as Q, K, V for self attention

        x = torch.squeeze(x, dim=1) # collapse singleton sequence length dimension before final linear layers

        x = self.FFN(x)

        # extra conditional for TallyNet to allow larger final outputs
        if (self.FINAL_ACTIVATION_FLAG == True):  
            x = self.FinalActivation(x)
        # end if

        
        return x

# end of Class def CNN_MHA_CLassifier











'''
Func: Create_TCCSNet_Classifier()
Purpose: Generates Pytorch Model with TCCSNet model for activity recognition
    INPUTS
sample_length - [int]
    Number of samples in a window
total_axes - [int]
    Number of features of the input time series
        
    OUTPUT
model  - PyTorch model using CNN-MHA for Human Actvitiy Recognition
'''

def Create_TCN_Classifier(sample_length, total_axes, num_tcn_layers):


    # Put together all blocks into model
    model =  TCN_Classifier(sample_length, total_axes, num_tcn_layers)

    return model

# end of def Create_TCCSNet_Classifier





'''
Func: Create_TCN_Seq2OneClassifier()
Purpose: Generates Pytorch Model with TCCSNet model for activity recognition
    INPUTS
sample_length - [int]
    Number of samples in a window
total_axes - [int]
    Number of features of the input time series
        
    OUTPUT
model  - PyTorch model using CNN-MHA for Human Actvitiy Recognition
'''

def Create_TCN_Seq2OneClassifier(total_axes, MODEL_ARCH_FLAG = 1):

    # fills exact input for center point to contain full receptive field
    if MODEL_ARCH_FLAG == 1: # Default for Paper 
        sample_length = 1023 # ~64 seconds @ 16 Hz
        num_tcn_layers = 9
        FINAL_ACTIVATION_FLAG = True
    elif MODEL_ARCH_FLAG == 2:
        sample_length = 511 # ~32 seconds @ 16 Hz
        num_tcn_layers = 8
        FINAL_ACTIVATION_FLAG = True
    elif MODEL_ARCH_FLAG == 3:
        sample_length = 255 # ~16 sec @ 16 Hz
        num_tcn_layers = 7
        FINAL_ACTIVATION_FLAG = True
    
    ### TALLYNET VERSIONS
    elif MODEL_ARCH_FLAG == 11: # Default for Paper 
        sample_length = 1023 # ~64 seconds @ 16 Hz
        num_tcn_layers = 9
        FINAL_ACTIVATION_FLAG = False
    elif MODEL_ARCH_FLAG == 12: # 15 Second @ 16 Hz For fair Comparison in TallyNet
        sample_length = 255
        num_tcn_layers = 7
        FINAL_ACTIVATION_FLAG = False

    else: # Default to Paper Dimensions
        print("INVALID MODEL_ARCH_FLAG INPUT. Defaulting to approximate paper standard with 1023 seq_len and 9 tcn layers.") 
        sample_length = 1023 # fills exact input for center point to contain full receptive field
        num_tcn_layers = 9
        FINAL_ACTIVATION_FLAG = True
    # end of switch MODEL_ARCH_FLAG

    # Put together all blocks into model
    model =  TCN_Seq2OneClassifier(sample_length, total_axes, num_tcn_layers, FINAL_ACTIVATION_FLAG)

    return model

# end of def Create_TCN_Seq2OneClassifier



    
