'''
File: Arch_ResNet_Model.py
Version: 0.2
Author: James Jolly
Purpose: 
	Allow for Keras Implementation of ResNet (Residual Architecture) within TensorFlow
	for use on IMU temporal data.
    Modifying ResNet10 model from CTC paper.
Version History:
	0.1 Development
    0.2 Renamed from "IMU_ResNetModel.py" to "Arch_ResNet_Model.py" 
    	to match other backbone files, and combined model call
        and model details into single file.
'''
# TensorFlow, keras, np
# import tensorflow as tf
# from tensorflow import keras



##### CODE FROM CTC REIMPLEMENTATION


"""1D ResNet CNN-LSTM Model for inertial data"""
"""
PyTorch implementation of 1D ResNet CNN-LSTM Model for inertial data
Converted from TensorFlow / Keras version
"""

"""Initial Conversion from CTC loss paper with OpenAI ChatGPT,
heavily modified and edited by James Jolly"""

import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_NORM_EPSILON = 1e-5


class Conv1DFixedPadding(nn.Module):
    """
    Strided 1D convolution with explicit padding (TensorFlow-style SAME padding)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if stride == 1:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding='same',
                bias=True
            )
        else: # if stride != 1
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding='valid',
                bias=True
            )
        # end of if stride == 1

    def forward(self, x):
        # x shape: (batch, channels, time)
        if self.stride > 1:
            pad_total = self.kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)
# end of Conv1DFixedPadding(nn.Module)

class ResBlock(nn.Module):
    """
    One residual CNN block
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, shortcut):
        super().__init__()
        self.shortcut = shortcut

        self.conv1 = Conv1DFixedPadding(
            in_channels, out_channels, kernel_size, stride
        )
        self.conv2 = Conv1DFixedPadding(
            out_channels, out_channels, kernel_size, stride=1
        )

        if self.shortcut:
            self.conv_sc = Conv1DFixedPadding(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_channels, eps=BATCH_NORM_EPSILON)
        self.bn2 = nn.BatchNorm1d(out_channels, eps=BATCH_NORM_EPSILON)

    def forward(self, x):
        shortcut = x
        if self.shortcut:
            shortcut = self.conv_sc(shortcut)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = x + shortcut
        x = self.relu(x)
        x = self.bn2(x)
        return x
# end of ResBlock(nn.Module)


class BlockLayer(nn.Module):
    """
    One layer consisting of multiple residual blocks
    """

    def __init__(self, num_blocks, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        blocks = []
        blocks.append(
            ResBlock(
                in_channels, out_channels, kernel_size, stride, shortcut=True
            )
        )

        for _ in range(num_blocks - 1):
            blocks.append(
                ResBlock(
                    out_channels, out_channels, kernel_size, stride=1, shortcut=False
                )
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
# end of BlockLayer(nn.Module)


class LSTMLayer(nn.Module):
    """
    One LSTM layer with optional residual connection
    """

    def __init__(self, input_size, hidden_size, shortcut):
        super().__init__()
        self.hidden_size = hidden_size
        self.shortcut = shortcut

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        if self.shortcut and input_size != hidden_size:
            self.proj = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        else:
            self.proj = None

    def forward(self, x):
        # x shape: (batch, time, features)
        shortcut = x

        x, _ = self.lstm(x)

        if self.shortcut:
            if self.proj is not None:
                # convert to (batch, channels, time) for conv
                sc = shortcut.transpose(1, 2)
                sc = self.proj(sc)
                sc = sc.transpose(1, 2)
            else:
                sc = shortcut

            x = x + sc

        return x
# end of def LSTMLayer(nn.Module)


class Model(nn.Module):
    """
    Residual CNN-LSTM Model
    """

    def __init__(self, num_classes, input_channels, input_length, specs):
        super().__init__()
        self.input_length = input_length
        self.seq_pool = specs["seq_pool"]
        self.block_specs = specs["block_specs"]
        self.lstm_specs = specs["lstm_specs"]

        self.conv1 = Conv1DFixedPadding(
            input_channels,
            specs["conv_1_filters"],
            specs["conv_1_kernel_size"],
            specs["conv_1_stride"]
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(
            specs["conv_1_filters"], eps=BATCH_NORM_EPSILON
        )

        self.conv_blocks = nn.ModuleList()
        in_channels = specs["conv_1_filters"]
        for blocks, filters, kernel_size, stride in self.block_specs:
            self.conv_blocks.append(
                BlockLayer(
                    blocks, in_channels, filters, kernel_size, stride
                )
            )
            in_channels = filters

        self.lstm_blocks = nn.ModuleList()
        lstm_input_size = in_channels
        for units, shortcut in self.lstm_specs:
            self.lstm_blocks.append(
                LSTMLayer(lstm_input_size, units, shortcut)
            )
            lstm_input_size = units

        self.classifier = nn.Linear(lstm_input_size, num_classes)

    def forward(self, x):
        # x shape: (batch, time, channels)
        # x = x.transpose(1, 2)  # (batch, channels, time)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        for block in self.conv_blocks:
            x = block(x)

        # back to (batch, time, features) for LSTM
        x = x.transpose(1, 2)

        for lstm in self.lstm_blocks:
            x = lstm(x)

        if True: # Only grab final layer since collapsing sequence to single point
            x = x[:, -1, :]
        # end of if Collapsing to final state

        x = self.classifier(x)
        return x

    def get_seq_length(self):
        return self.input_length // self.seq_pool

    def get_seq_pool(self):
        return self.seq_pool

    def get_out_pool(self):
        return self.seq_pool
# end of class Model()



# Function: IMU_ResNet_Model
# Purpose: Return Keras ResNet model to use on IMU data
# Inputs: 
# Outputs: 

# Hyperparameters
L2_LAMBDA = 1e-5
LR_BOUNDARIES = [5, 10, 15]
LR_VALUE_DIV = [1., 10., 100., 1000.]
LR_DECAY_STEPS = 1


def IMU_ResNet_Model(input_shape, output, dataset="clemson"):
	
	l2_lambda=L2_LAMBDA
	input_length = input_shape[0]
	input_channels = input_shape[1]
	if dataset == "clemson":
		specs = {
			"seq_pool": 2,
			"conv_1_filters": 64,
			"conv_1_kernel_size": 1,
			"conv_1_stride": 1,
			"block_specs": [(1, 64, 3, 1), (1, 128, 3, 1), (1, 256, 5, 2),
				(1, 512, 5, 1)],
			"lstm_specs": [(64, False)]
		}
	num_classes=1 # adjust number of classes to be single regression value
	
	
	# print("input_length = {}".format(input_length))
	

	model = torch_inert_resnet_cnn_lstm.Model(num_classes=num_classes,
            input_channels = input_channels,                          
    		input_length=input_length, specs=specs)

	return model

# end of IMU_Resnet_Model








