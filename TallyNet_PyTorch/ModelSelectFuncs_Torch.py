'''
File: ModelSelectFuncs.py
Version: 4.0
Author: James Jolly
Purpose: 
    Allow for flexible Model Architecture layers inside TallyNet
Version History:
    4.0 - Added Reimplementation Backbones as 4XYZ model arch flags,
            where X is the paper it comes from, Y is alterate moels from the paper,
            and Z is me adjusting and tuning layer features (depth, filters, etc.)
    3.1 - Changed LSTM layers to be marked as "Batch_first=false" to match CNN dimension ordering 
    3.0 - Huge sweep across layer parameters with architectures with number "1XYZ", 
            where X is conv filter count, y is LSTM node code, 
            and Z is the # of hidden nodes in dense layer 
    2.0 - Simplification of Models to best performing 101 and 118 from search
    1.0 - Intial creation of file
'''
# TensorFlow, keras, np
# import tensorflow as tf
# from tensorflow import keras
# from IMU_ResNet_Model import IMU_ResNet_Model
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from Arch_ResNet_Model import IMU_ResNet_Model
from Arch_CNN_MHA_Classifier import Create_CNN_MHA
from Arch_TCCSNet_Classifier import Create_CSNet_Classifier
from Arch_TCCSNet_Classifier import Create_TCCSNet_Classifier
from Arch_TCN_Classifier import Create_TCN_Classifier
from Arch_TCN_Classifier import Create_TCN_Seq2OneClassifier
from Arch_GRU_Classifier import Create_GRU_Classifier
from Arch_CustomGRU_Classifier import Create_CustomGRU_Classifier

# Function: ModelSelect
# Purpose: Seperated function to have a separate space for model architectures to reduce the 
#          number of lines in the Training File (since many architectures are lenghty code) 
# Inputs: MODEL_ARCH_FLAG: switch value (int) that selects which model to use
# Outputs: model: the keras model used for training
#          NUM_EPOCHS: the number of epochs used to train this model (sometimes reduced for 
#                     longer architectures to save training time)




# Define the generic EMNIST classification architecture
# The `feature_extractor` argument takes images as input and produces feature vectors.
# These vectors can be of any length. 
class TallyNetClassifier(nn.Module):
    def __init__(self, feature_extractor, recurrent_block, regression_block):
        super(TallyNetClassifier, self).__init__()

        # feature encoder
        self.feature_extractor = feature_extractor

        self.recurrent_block = recurrent_block

        self.regression_block = regression_block


    def forward(self, x):
        x = self.feature_extractor(x) 
        x = x.transpose(1, 2) # flip [N, C, Len] from Conv to [N, Len, C] for LSTM
        x,(hidden_state, cell_state) = self.recurrent_block(x)
        x = self.regression_block(x)
        return x

# end of TallyNetClassifier

def ModelSelect_Torch(MODEL_ARCH_FLAG, sample_length, total_axes, ):

    # Reimplmentation Backbones Adapted to TallyNet
    if (4000 <= MODEL_ARCH_FLAG and MODEL_ARCH_FLAG < 5000):

        ##### CNN-MHA ARCHITECTURES #####
        if MODEL_ARCH_FLAG == 4001: # if CNN-MHA
            NUM_EPOCHS=25
            # Create_CNN_MHA(sample_length, total_axes):
            model = Create_CNN_MHA(sample_length, total_axes)

        ##### TCCSNet and CSNet ARCHITECTURES #####
        elif MODEL_ARCH_FLAG == 4101: # if CSNET
            NUM_EPOCHS=25
            # Create_CSNet_Classifier(sample_length, total_axes):
            model = Create_CSNet_Classifier(sample_length, total_axes)
        elif MODEL_ARCH_FLAG == 4111: # if TCCSNET
            NUM_EPOCHS=25
            # Create_TCCSNet_Classifier(sample_length, total_axes)
            model = Create_TCCSNet_Classifier(sample_length, total_axes)

        ##### TCN-MHA ARCHITECTURES #####
        elif MODEL_ARCH_FLAG == 4201: # if Eating Speed TCN-MHA
            NUM_EPOCHS=25
            # Create_TCN_Classifier(sample_length, total_axes, num_tcn_layers)
            num_tcn_layers = 6 # limted levels of dilation 
            model = Create_TCN_Classifier(sample_length, total_axes, num_tcn_layers)
        elif MODEL_ARCH_FLAG == 4211: # if Eating Speed TCN-MHA
            NUM_EPOCHS=25
            # Create_TCN_Seq2OneClassifier(total_axes, HARD-CODED Value)
            TCN_Option = 11 # original paper, but without sigmoid to allow for large integer output 
            model = Create_TCN_Seq2OneClassifier(total_axes, MODEL_ARCH_FLAG = TCN_Option)
        elif MODEL_ARCH_FLAG == 4212: # if Eating Speed TCN-MHA
            NUM_EPOCHS=25
            # Create_TCN_Seq2OneClassifier(total_axes, HARD-CODED Value)
            TCN_Option = 12 # 15sec @ 16 Hz to match other TallyNet Sizes 
            model = Create_TCN_Seq2OneClassifier(total_axes, MODEL_ARCH_FLAG = TCN_Option)

        ##### CNN_MHA_GRU ARCHITECTURES AND MODIFICATIONS #####
        elif MODEL_ARCH_FLAG == 4301: # if CNN-GRU Model # Standard
            NUM_EPOCHS=25
            # model = Create_GRU_Classifier(sample_length, total_axes, (opt) MODEL_ARCH=0)
            model = Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 100) # use default model architecture without final activation
        elif MODEL_ARCH_FLAG == 4302: # if CNN-GRU Model # Smaller size
            NUM_EPOCHS=25
            model = Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 101) 
        elif MODEL_ARCH_FLAG == 4311: # if CNN-GRU Model # Using 3 cnn blocks 
            NUM_EPOCHS=25
            model = Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 200) 
        elif MODEL_ARCH_FLAG == 4321: # if CNN-GRU Model # Using 5 cnn blocks
            NUM_EPOCHS=25
            model = Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 300)
        elif MODEL_ARCH_FLAG == 4331: # if CNN-GRU Model # Removing MHA block
            NUM_EPOCHS=25
            model = Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 401) 
        elif MODEL_ARCH_FLAG == 4332: # if CNN-GRU Model # Removing both GRU
            NUM_EPOCHS=25
            model = Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 402) 
        elif MODEL_ARCH_FLAG == 4333: # if CNN-GRU Model # Removing second GRU (keeping first)
            NUM_EPOCHS=25
            model = Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 403) 
        elif MODEL_ARCH_FLAG == 4334: # if CNN-GRU Model # Removing second GRU (keeping first)
            NUM_EPOCHS=25
            model = Create_GRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 404) 

        ##### CUSTOMIZATIONS TO CNN_MHA_GRU ARCHITECTURES #####
        elif MODEL_ARCH_FLAG == 4901: # if CNN-GRU Model # Removing MHA block # adding 3 CNN Blocks
            NUM_EPOCHS=25
            model = Create_CustomGRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 0) 
        elif MODEL_ARCH_FLAG == 4902: # if CNN-GRU Model # Moving MHA block before first GRU # Adding 3 CNN Blocks 
            NUM_EPOCHS=25
            model = Create_CustomGRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 1) 
        elif MODEL_ARCH_FLAG == 4903: # if CNN-GRU Model # Removing MHA block # adding 3 CNN Blocks # only 1 max pool after 2nd cnn layer  
            NUM_EPOCHS=25
            model = Create_CustomGRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 3) 
        elif MODEL_ARCH_FLAG == 4904: # if CNN-GRU Model # Removing MHA block # adding 3 CNN Blocks # No Max Pooling
            NUM_EPOCHS=25
            model = Create_CustomGRU_Classifier(sample_length, total_axes, MODEL_ARCH_FLAG = 4) 
        
            
        else:
            raise ValueError("Invalid MODEL_ARCH_FLAG passed into ModelSelect_Torch()")
        # end of Reimplmentation Backbones switch
        

    # Test Hard coded A1333 Keras model as Pytorch Model
    # Hard Coded LSTM input length for 15 second 15 Hz data
    if (3000 <= MODEL_ARCH_FLAG and MODEL_ARCH_FLAG < 4000):

        dropoutRate = 0.2
        BaseFilterCnt = 24
        LstmNodeCnt = 24
        DenseNeuronCnt = 24

        # Output Classes
        num_outputs = 1 # Single value intake or not
        # num_outputs = 2 # predict bites and drinks separately

        Arch1333Feature = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(in_channels=total_axes,
                      out_channels=BaseFilterCnt,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(p=dropoutRate),

            nn.Conv1d(in_channels=BaseFilterCnt,
                      out_channels=BaseFilterCnt,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(p=dropoutRate),

            nn.MaxPool1d(2),


            # Conv Block 2
            nn.Conv1d(in_channels=BaseFilterCnt,
                      out_channels=BaseFilterCnt*2,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(p=dropoutRate),

            nn.Conv1d(in_channels=BaseFilterCnt*2,
                      out_channels=BaseFilterCnt*2,
                      kernel_size=3,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(p=dropoutRate),

            nn.MaxPool1d(2),

            # Conv Block 3
            nn.Conv1d(in_channels=BaseFilterCnt*2,
                      out_channels=BaseFilterCnt*3,
                      kernel_size=5,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(p=dropoutRate),

            nn.Conv1d(in_channels=BaseFilterCnt*3,
                      out_channels=BaseFilterCnt*3,
                      kernel_size=5,
                      stride=1),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(p=dropoutRate)
        )



        Arch1333RecurrenceForward = nn.Sequential(
            nn.LSTM(45, LstmNodeCnt,
                    num_layers = 2,
                    batch_first=True,
                    # dropout = dropoutRate ### Pottentially Add later
                    bidirectional = False)
        )

        Arch1333RecurrenceBidirect = nn.Sequential(
            nn.LSTM(45, LstmNodeCnt,
                    num_layers = 2,
                    batch_first=True,
                    # dropout = dropoutRate ### Potentially Add later
                    bidirectional = True)
        )

        Arch1333Regression = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropoutRate),
            nn.LazyLinear(DenseNeuronCnt),
            nn.Dropout(p=dropoutRate),
            nn.Linear(DenseNeuronCnt, num_outputs)
        )



        if MODEL_ARCH_FLAG == 3001: # A1333 Torch Unidirectional LSTM
            model = TallyNetClassifier(Arch1333Feature, Arch1333RecurrenceForward, Arch1333Regression)
        if MODEL_ARCH_FLAG == 3002: # A1333 Torch Bidirectional LSTM
            model = TallyNetClassifier(Arch1333Feature, Arch1333RecurrenceBidirect, Arch1333Regression)
        # end of switch

        NUM_EPOCHS = 25

        # print("Finished generating model architecture.")



    # end of if PyTorch

    # use 2XXX for additional common architectures
    if (2000 <= MODEL_ARCH_FLAG and MODEL_ARCH_FLAG < 3000):
        if MODEL_ARCH_FLAG == 2001: # ResNet50
            NUM_EPOCHS=10 # Model Takes a while, so decrease epochs for exploratory testing
            num_outputs = 1 # using regression

            model = IMU_ResNet_Model([sample_length, total_axes],num_outputs)
        # end if ResNet50

        if MODEL_ARCH_FLAG == 2002: # ResNet50
            NUM_EPOCHS=5 # Model Takes a while, so decrease epochs for exploratory testing

        # end if ResNet50

        if MODEL_ARCH_FLAG == 2003: # ResNet50
            NUM_EPOCHS=5 # Model Takes a while, so decrease epochs for exploratory testing

        # end if ResNet50



    # end if 2XXX Architectures

    # Use 1 to signify first gen search parameters, and each digit to signify a scan variable.
    if (1000 <= MODEL_ARCH_FLAG and MODEL_ARCH_FLAG < 2000):
        print("Arch {:d}".format(MODEL_ARCH_FLAG))
        NUM_EPOCHS=50 # Model Takes a while, so decrease epochs for exploratory testing

        print("Architecture Scan of A300 architecture.")
        print("Added Dropout layers to model.")

        FilterIdx = int((MODEL_ARCH_FLAG / 100) % 10) # Hundreds Digit
        LstmIdx = int((MODEL_ARCH_FLAG / 10) % 10) # Tens Digit
        DenseIdx = int(MODEL_ARCH_FLAG % 10) # Ones Digit


        BaseFilterOptions = [8, 12, 16, 24, 32, 40, 48, 64]
        BaseFilterCnt = BaseFilterOptions[FilterIdx]

        LstmNodesOptions = [8, 12, 16, 24, 32, 40, 48, 64]
        LstmNodeCnt = BaseFilterOptions[LstmIdx]

        DenseNeuronOptions = [8, 12, 16, 24, 32, 40, 48, 64]
        DenseNeuronCnt = DenseNeuronOptions[DenseIdx]

        print("Using {:d} Filters as base in ConvKernels".format(BaseFilterCnt))
        print("Using {:d} Nodes in LSTM layers".format(LstmNodeCnt))
        print("Using {:d} Neurons in Dense Hidden Layer".format(DenseNeuronCnt))

        model = keras.Sequential([
            keras.layers.Conv1D(input_shape=(sample_length,total_axes,),
                filters=BaseFilterCnt,
                kernel_size=3, #initial layer 1/4 of a second
                strides=1,
                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Conv1D(filters=BaseFilterCnt,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),


            keras.layers.MaxPool1D(),


            keras.layers.Conv1D(filters=2*BaseFilterCnt,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Conv1D(filters=2*BaseFilterCnt,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),


            keras.layers.MaxPool1D(),


            keras.layers.Conv1D(filters=3*BaseFilterCnt,
                kernel_size=5,
                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Conv1D(filters=3*BaseFilterCnt,
                kernel_size=5,
                activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),



        #    keras.layers.SimpleRNN(units=128,activation='relu'),
        #    keras.layers.SimpleRNN(units=128,activation="relu",return_sequences=True),

            ###  SECOND PHASE: TIME MEMORY OF GESTURES  ### 
            #####  TAKEN FROM OREBA REIMPLEMENTATION  #####
            keras.layers.LSTM(LstmNodeCnt, return_sequences=True,
                activation="tanh",recurrent_activation="hard_sigmoid"),
            keras.layers.LSTM(LstmNodeCnt, 
                activation="tanh",recurrent_activation="hard_sigmoid"),
            ###  END OF BORROWED REIMPLEMENTATION  ###

            keras.layers.Flatten(),  # must flatten to feed dense layer
            keras.layers.Dropout(0.2),
            keras.layers.Dense(DenseNeuronCnt),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
            ])

    elif (MODEL_ARCH_FLAG==101):
        print("Arch 101")
        NUM_EPOCHS=25 # Model Takes a while, so decrease epochs for exploratory testing

        print("Batch Norm Layers, Max Pooling, & Separable with Kernel Size Adjustments")
        print("LSTM Added at End of CNN")
        QtrSecNumPts = 4 # Used with 15 or 16 Hz Data to cover several seconds by final Conv layer
        print("QtrSecNumPts = {}".format(QtrSecNumPts))

        model = keras.Sequential([
            keras.layers.SeparableConv1D(input_shape=(sample_length,total_axes,),
                filters=50,
                kernel_size=QtrSecNumPts, #initial layer 1/4 of a second
                strides=1,
                activation='relu'),
            keras.layers.BatchNormalization(),  ### ADDITION FROM BASELINE ###

            keras.layers.SeparableConv1D(filters=50,
                kernel_size=QtrSecNumPts,
                activation='relu'),
            keras.layers.BatchNormalization(),  ### ADDITION FROM BASELINE ###

            keras.layers.MaxPool1D(),

            keras.layers.SeparableConv1D(filters=60,
                kernel_size=QtrSecNumPts,
                activation='relu'),
            keras.layers.BatchNormalization(),  ### ADDITION FROM BASELINE ###

            keras.layers.SeparableConv1D(filters=70,
                kernel_size=QtrSecNumPts,
                activation='relu'),
            keras.layers.BatchNormalization(),  ### ADDITION FROM BASELINE ###

            keras.layers.SeparableConv1D(filters=70,
                kernel_size=QtrSecNumPts,
                activation='relu'),
            keras.layers.BatchNormalization(),  ### ADDITION FROM BASELINE ###

            keras.layers.MaxPool1D(),

            keras.layers.SeparableConv1D(filters=80,
                kernel_size=QtrSecNumPts,
                activation='relu'),
            keras.layers.BatchNormalization(),  ### ADDITION FROM BASELINE ###

            keras.layers.SeparableConv1D(filters=80,
                kernel_size=QtrSecNumPts,
                activation='relu'),


        #    keras.layers.SimpleRNN(units=128,activation='relu'),
        #    keras.layers.SimpleRNN(units=128,activation="relu",return_sequences=True),

            ###  SECOND PHASE: TIME MEMORY OF GESTURES  ### 
            #####  TAKEN FROM OREBA REIMPLEMENTATION  #####
            keras.layers.LSTM(64, return_sequences=True,
                activation="tanh",recurrent_activation="hard_sigmoid"),
            keras.layers.LSTM(64, 
                activation="tanh",recurrent_activation="hard_sigmoid"),
            ###  END OF BORROWED REIMPLEMENTATION  ###

            keras.layers.Flatten(),  # must flatten to feed dense layer
            keras.layers.Dense(1)
            ])



    elif (MODEL_ARCH_FLAG==118):
        print("Arch 118")
        NUM_EPOCHS=25 # Model Takes a while, so decrease epochs for exploratory testing

        print("Batch Norm Layers, Max Pooling, & Separable with Kernel Size Adjustments")
        print("2 LSTM Added at End of CNN")
        print("NEW to 118:")
        print("from 116: Further reduced filter size to non-powers of two, AND reduced LSTM gates slightly.")

        model = keras.Sequential([
            keras.layers.SeparableConv1D(input_shape=(sample_length,total_axes,),
                filters=24,
                kernel_size=3, #initial layer 1/4 of a second
                strides=1,
                activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.SeparableConv1D(filters=24,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),


            keras.layers.MaxPool1D(),


            keras.layers.SeparableConv1D(filters=32,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.SeparableConv1D(filters=32,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),


            keras.layers.MaxPool1D(),


            keras.layers.SeparableConv1D(filters=40,
                kernel_size=5,
                activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.SeparableConv1D(filters=40,
                kernel_size=5,
                activation='relu'),
            keras.layers.BatchNormalization(),



        #    keras.layers.SimpleRNN(units=128,activation='relu'),
        #    keras.layers.SimpleRNN(units=128,activation="relu",return_sequences=True),

            ###  SECOND PHASE: TIME MEMORY OF GESTURES  ### 
            #####  TAKEN FROM OREBA REIMPLEMENTATION  #####
            keras.layers.LSTM(24, return_sequences=True,
                activation="tanh",recurrent_activation="hard_sigmoid"),
            keras.layers.LSTM(24, 
                activation="tanh",recurrent_activation="hard_sigmoid"),
            ###  END OF BORROWED REIMPLEMENTATION  ###

            keras.layers.Flatten(),  # must flatten to feed dense layer
            keras.layers.Dense(1)
            ])


    elif (MODEL_ARCH_FLAG==300):
        print("Arch 300")
        NUM_EPOCHS=25 # Model Takes a while, so decrease epochs for exploratory testing

        print("Batch Norm Layers, Max Pooling, & Separable with Kernel Size Adjustments")
        print("2 LSTM Added at End of CNN")
        print("NEW to 118:")
        print("from 118: TwoDenseLayers at the end, with 24 hidden neurons.")

        model = keras.Sequential([
            keras.layers.SeparableConv1D(input_shape=(sample_length,total_axes,),
                filters=24,
                kernel_size=3, #initial layer 1/4 of a second
                strides=1,
                activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.SeparableConv1D(filters=24,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),


            keras.layers.MaxPool1D(),


            keras.layers.SeparableConv1D(filters=32,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.SeparableConv1D(filters=32,
                kernel_size=3,
                activation='relu'),
            keras.layers.BatchNormalization(),


            keras.layers.MaxPool1D(),


            keras.layers.SeparableConv1D(filters=40,
                kernel_size=5,
                activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.SeparableConv1D(filters=40,
                kernel_size=5,
                activation='relu'),
            keras.layers.BatchNormalization(),



        #    keras.layers.SimpleRNN(units=128,activation='relu'),
        #    keras.layers.SimpleRNN(units=128,activation="relu",return_sequences=True),

            ###  SECOND PHASE: TIME MEMORY OF GESTURES  ### 
            #####  TAKEN FROM OREBA REIMPLEMENTATION  #####
            keras.layers.LSTM(24, return_sequences=True,
                activation="tanh",recurrent_activation="hard_sigmoid"),
            keras.layers.LSTM(24, 
                activation="tanh",recurrent_activation="hard_sigmoid"),
            ###  END OF BORROWED REIMPLEMENTATION  ###

            keras.layers.Flatten(),  # must flatten to feed dense layer
            keras.layers.Dense(24),
            keras.layers.Dense(1)
            ])
    # end of Model Architecture Switch Statement


    return model, NUM_EPOCHS

# end of ModelSelect() Function
