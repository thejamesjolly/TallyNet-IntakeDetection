'''
Filename: TCCSNet_Classifier.py
Author: James Jolly
Purpose: Implement PyTorch model of the Human Activity Recognition from IMU data found in the paper,
        "Temporal-channel convolution with self-attention network for juman activity recognition using wearable sensors" by Ehab Esaa and Islam R. Abdelmaksoud
Version: 
    0.1: Intial Code
'''

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR





class ConvBlock(nn.Module):
    """
    CNN block used throughout TCCSnet architecture

    Expects inputs to be of order [Batch_size, Features, Seq_Length]
    """

    def __init__(self, in_channels, filter1, kernel1, filter2, kernel2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, filter1, kernel1)
        self.conv2 = nn.Conv1d(filter1, filter2, kernel2)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(filter1)
        self.bn2 = nn.BatchNorm1d(filter2)
        self.maxpool = nn.MaxPool1d(2)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool(x)
        
        return x
# end of def class ConvBlock()



#
class SelfAttenBlock(nn.Module):
    """
    Self Attention Block used in TCCSnet architecture

    Expects inputs to be of order [Batch_size, Seq_Length, Features]
    """

    def __init__(self, d_head, num_heads, drop_rate):
        super().__init__()

        # standard in paper
        # d_k = 128
        # num_heads = 2
        # d_model = 256

        self.d_head = d_head # dq = dk = dimension of Query and Keys
        self.num_heads = num_heads
        self.d_model = d_head * num_heads
        self.d_ff = 32 # number of hidden features in Feed Forward Network
    
        
        
        self.LayerNorm = nn.LayerNorm([self.d_model])
        
    
        self.MHA_Block = nn.MultiheadAttention(self.d_model, self.num_heads, batch_first = True)


        self.Dropout = nn.Dropout(p=drop_rate)

        # Define Feed Forward Network
        self.FFN = nn.Sequential(
            #
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.ReLU()

        )

    def forward(self, x):

        # Pass data through MHA with residual
        res1 = x.clone() # Save residual 
        x = self.LayerNorm(x) # Normalize over channels
        x, _ = self.MHA_Block(x, x, x) # Pass all input as Q, K, V for self attention  
        x = self.Dropout(x)
        x = x + res1 # Add residual back

        # Pass Data trhrough FNN with residual
        res2 = x.clone() # save 2nd residual 
        x = self.LayerNorm(x)
        x = self.FFN(x)
        x = x + res2
                
        return x
# end of def class SelfAttenBlock()




class AddPosEncodingBlock(nn.Module):
    """
    Adds Positional Encoding to data in TCCSnet architecture

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




class WeightedConcat(nn.Module):
    """
    Merge two tensors with weighted value used throughout TCCSnet architecture
    """

    def __init__(self, initial_weight = 0.5):
        super().__init__()
        
        self.weight = nn.Parameter(torch.tensor(initial_weight))
        

    def forward(self, x1, x2):
        
        x1 = x1 * self.weight
        x2 = x2 * (1 - self.weight)
        x = torch.cat((x1, x2), dim=1)
        return x
# end of def class ConvBlock()




# Define the CSNet classification architecture
# These vectors can be of any length. 
class CSNet_Classifier(nn.Module):
    def __init__(self, sample_length, total_axes):
        super(CSNet_Classifier, self).__init__()

        self.num_classes = 1 # final output is only prediciting intake or not

        self.drop_rate = 0.4 

        self.d_head = 128
        self.num_heads = 2
        self.d_model = self.d_head * self.num_heads
        self.d_ff_final = 512 # Num hidden neurons in final FFN block

        self.in_channels = total_axes

        # Blocks
        self.Conv1 = ConvBlock(self.in_channels, 32, 3, 64, 3) # Time

        self.drop1 = nn.Dropout(p=self.drop_rate) # Time

        # Project to d_model
        self.MHA_Prep1 = nn.Linear(64, self.d_model) # Time

        # max length after max pool is less than half of sample_length
        self.PosEncode = AddPosEncodingBlock(self.d_model, round(sample_length/2))

        self.SelfAtten1 = SelfAttenBlock(self.d_head, self.num_heads, self.drop_rate) # Time
        self.SelfAtten2 = SelfAttenBlock(self.d_head, self.num_heads, self.drop_rate) # Time
    
        # Conv Block

        self.Merge = WeightedConcat()
        
        self.Conv3 = ConvBlock(self.d_model, 64, 3, 64, 3) # Joint
        
        # Final Dense Layer of shape [BatchSize, Neurons]
        self.FFN = nn.Sequential(
            nn.LazyLinear(self.d_ff_final),
            nn.Dropout(p=self.drop_rate)
        )
        self.OutputLayer = nn.Linear(self.d_ff_final, self.num_classes)

        

    def forward(self, x):

        
        ##############################
        ###### TIME_WISE BRANCH ######
        ##############################
        x = self.Conv1(x) 
        # print(x.size())
        x = self.drop1(x)

        # print(x.size())

        # Manage dimensions to match PosEncoder and MHA expected input 
        # [Batch_size, Features, Seq_Length] -> [Batch_size, Seq_Length, Features]
        x = x.transpose(1, 2)
        
        x = self.MHA_Prep1(x) # Project to d_model
        
        x = self.PosEncode(x)
        x = self.SelfAtten1(x) # Pass all input as Q, K, V for self attention
        x = self.SelfAtten2(x) # Pass all input as Q, K, V for self attention
        

        # Swap dimension to Convolution expected order 
        # [Batch_size, Features, Seq_Length] -> [Batch_size, Seq_Length, Features]
        x = x.transpose(1, 2)
        x = self.Conv3(x)
        x = nn.Flatten()(x)
        x = self.FFN(x)
        x = self.OutputLayer(x)

        return x

# end of Class def CSNet_Classifier



# Define the TCCSnet classification architecture
# The `feature_extractor` argument takes images as input and produces feature vectors.
# These vectors can be of any length. 
class TCCSNet_Classifier(nn.Module):
    def __init__(self, sample_length, total_axes):
        super(TCCSNet_Classifier, self).__init__()

        self.num_classes = 1 # final output is only prediciting intake or not

        self.drop_rate = 0.4 

        self.d_head = 128
        self.num_heads = 2
        self.d_model = self.d_head * self.num_heads
        self.d_ff_final = 512 # Num hidden neurons in final FFN block

        self.in_channels = total_axes
        self.in_seq_len = sample_length

        # Blocks
        self.Conv1 = ConvBlock(self.in_channels, 32, 3, 64, 3) # Time
        self.Conv2 = ConvBlock(self.in_seq_len, 32, 3, 64, 3) # Channel

        self.drop1 = nn.Dropout(p=self.drop_rate) # Time
        self.drop2 = nn.Dropout(p=self.drop_rate) # Channel

        # Project to d_model
        self.MHA_Prep1 = nn.Linear(64, self.d_model) # Time
        self.MHA_Prep2 = nn.Linear(64, self.d_model) # Channel

        # max length after max pool is less than half of sample_length
        self.PosEncode = AddPosEncodingBlock(self.d_model, round(sample_length/2))

        self.SelfAtten1 = SelfAttenBlock(self.d_head, self.num_heads, self.drop_rate) # Time
        self.SelfAtten2 = SelfAttenBlock(self.d_head, self.num_heads, self.drop_rate) # Time
        self.SelfAtten3 = SelfAttenBlock(self.d_head, self.num_heads, self.drop_rate) # Channel
        self.SelfAtten4 = SelfAttenBlock(self.d_head, self.num_heads, self.drop_rate) # Channel
    
        # Conv Block

        self.Merge = WeightedConcat()
        
        self.Conv3 = ConvBlock(self.d_model, 64, 3, 64, 3) # Joint
        
        # Final Dense Layer of shape [BatchSize, Neurons]
        self.FFN = nn.Sequential(
            nn.LazyLinear(self.d_ff_final),
            nn.Dropout(p=self.drop_rate)
        )
        self.OutputLayer = nn.Linear(self.d_ff_final, self.num_classes)

        

    def forward(self, x):

        # Split input into two branches
        x_c = x.clone()
        
        ##############################
        ###### TIME_WISE BRANCH ######
        ##############################
        x = self.Conv1(x) 
        x = self.drop1(x)

        # Manage dimensions to match PosEncoder and MHA expected input 
        # [Batch_size, Features, Seq_Length] -> [Batch_size, Seq_Length, Features]
        x = x.transpose(1, 2)
        
        x = self.MHA_Prep1(x) # Project to d_model
        x = self.PosEncode(x)
        x = self.SelfAtten1(x) # Pass all input as Q, K, V for self attention
        x = self.SelfAtten2(x) # Pass all input as Q, K, V for self attention

        # output_size_time = [N, (seq_len-4)/2, 256]
        # output_size_channel = [N, (num_axes-4)/2, 256]
        

        #################################
        ###### CHANNEL WISE BRANCH ######
        #################################
        # Swap dimension around so that "sensor axes" are the "sequence" for conv and SelfAttention
        # [Batch_size, Features, Seq_Length] -> [Batch_size, Seq_Length, Features]
        x_c = x_c.transpose(1, 2)
        
        x_c = self.Conv2(x_c)
        x_c = self.drop2(x_c)

        # [Batch_size, Features, Channel_Seq_Length] -> [Batch_size, Channel_Seq_Length, Features]
        x_c = x_c.transpose(1, 2)
        
        x_c = self.MHA_Prep2(x_c) # Project to d_model
        x_c = self.SelfAtten3(x_c)  # Pass all input as Q, K, V for self attention
        x_c = self.SelfAtten4(x_c)  # Pass all input as Q, K, V for self attention

        ###########################
        ###### COMBINED PATH ######
        ###########################
        x = self.Merge(x, x_c)
        # Swap dimension to Convolution expected order 
        # [Batch_size, Features, Seq_Length] -> [Batch_size, Seq_Length, Features]
        x = x.transpose(1, 2)
        x = self.Conv3(x)
        x = nn.Flatten()(x)
        x = self.FFN(x)
        x = self.OutputLayer(x)

        return x

# end of Class def CNN_MHA_CLassifier




'''
Func: Create_CSNet_Classifier()
Purpose: Generates Pytorch Model with CSNet model for activity recognition
    INPUTS
sample_length - [int]
    Number of samples in a window
total_axes - [int]
    Number of features of the input time series
        
    OUTPUT
model  - PyTorch model using CNN-MHA for Human Actvitiy Recognition
'''

def Create_CSNet_Classifier(sample_length, total_axes):


    # Put together all blocks into model
    model =  CSNet_Classifier(sample_length, total_axes)

    return model

# end of def Create_CSNet_Classifier


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

def Create_TCCSNet_Classifier(sample_length, total_axes):


    # Put together all blocks into model
    model =  TCCSNet_Classifier(sample_length, total_axes)

    return model

# end of def Create_TCCSNet_Classifier

    