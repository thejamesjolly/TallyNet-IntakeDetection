'''
Filename: train-model.py
Original Code: Dr. Adam Hoover (train-model-5.py)
Edited By: James Jolly

Record

IMPROVEMENTS TO MAKE:
- Add Checkpointing to training based on best validation score

'''




# Library imports
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from PIL import Image
import math
import random


import numpy as np
import time
import sys

from io import UnsupportedOperation
import pickle as pkl


# Add shared location for auxillary functions
sys.path.insert(1, './../AuxillaryFunctions')

# import personal Functions
from GenerateClassDataFuncs import GenerateDataAndClasses_ClemCafe_WinCnt
from GenerateClassDataFuncs import GenerateDataAndClasses_OREBA_WinCnt

from ModelSelectFuncs_Torch import ModelSelect_Torch

# set random seed for reproducibility
print("Finished Loaded Libraries.")




# Global Variables for Pytorch Functions
# save locations
# best not to use your home directory for reading/writing large temporary files
# /scratch is much faster!
# data_dir = f"/scratch/{os.environ['USER']}/data"
# model_path = f"/scratch/{os.environ['USER']}/model.pt"

# Model and Training
batch_size=128 #input batch size for training (default: 64)
test_batch_size=1000 #input batch size for testing (default: 1000)
num_workers=9 # parallel data loading to speed things up
lr=0.001 # learning rate 
epsilon=1e-07 # learning rate in Adam Optimizer
gamma=0.7 # Learning rate step gamma (default: 0.7)
no_cuda=False #disables CUDA training (default: False)
save_model=True #save the trained model (default: False)
validation_split = 0.05 # percent of training data to withhold as validation
log_interval=500 #how many batches to wait before logging training status (default: 10) 
checkpoint_interval = 5 # number of epochs to save checkpoint

# additional derived settings
use_cuda = not no_cuda and torch.cuda.is_available()
pytorch_seed = random.randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff)
_ = torch.manual_seed(pytorch_seed)
print("For reproducibility: Pytorch seed is {}".format(pytorch_seed))

device = torch.device("cuda" if use_cuda else "cpu")

print("Device:", device)

print("Finished setting up PyTorch variables.")




# Training and testing functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output.squeeze(dim=1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('\r\tTrain epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), end='')
            
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.squeeze(dim=1)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)

    print('\nTest epoch {}: Average loss: {:.4f}'.format(
        epoch,
        test_loss), flush=True)
    
def train_and_test(model, save_name="default_saved_model.pt", epochs=5):
    # Train the linear model
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=epsilon)
    # scheduler = StepLR(optimizer, step_size=1, gamma=gamma) # Remove scheduler from CCIT RCDE Tuturoial code

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, valid_loader, epoch)
        # scheduler.step()
        
        if (epoch % checkpoint_interval == 0):
            print("Saving Checkpoint model at epoch {}...".format(epoch))
            checkpoint_name = save_name + "chkpt.pt"
            torch.save(model.state_dict(), checkpoint_name)

    if save_model: # Used to toggle in interactive
        final_name = save_name + "_final.pt"
        print("Saving model at filename: {}".format(final_name))
        torch.save(model.state_dict(), final_name)

def get_n_params(model): return sum(p.numel() for p in model.parameters())

print("Finished Defining training and validation functions.")


















if __name__ == '__main__':

    RESAMPLE_FLAG_FREQ = int(sys.argv[9]) # zero if keeping original data frequency, new freq otherwise

    #define constants for the data
    MODEL_NAME = sys.argv[1]
    CUT_sec = float(sys.argv[2]) # the total length of the window in [sec]
    STRIDE_sec = float(sys.argv[3]) # The time to move between each window when making examples in [sec]
    DATABASE_FLAG = int(sys.argv[4])
    RR_FLAG=int(sys.argv[5]) # 1 if using striped fold division, 0 if using block fold division
    FOLDS_TOTAL = int(sys.argv[7])
    FOLD_INDEX = int(sys.argv[6])%FOLDS_TOTAL # wrap data into valid fold number in case fold 
                    # index is given as [1,FOLDS_TOTAL] instead of [0,FOLDS_TOTAL)  

    MODEL_ARCH_FLAG = int(sys.argv[8]) # Value to edit hyper parameters in model architecture
        # 0 = default; 10 filters 2sec wide 1 sec stride, 10 filters 3data wide
        # 1 = Double number of filters on both levels to 20
        # 2 = Quadruple number of filters on both levels
        # 3 = Double first layer, quad second deeper layer


    if False: #eventually edit to include DOWN SAMPLING OPTIONS
        end_input_index=9
        DS_FLAG = int(sys.argv[end_input_index+1]) # 1 = linear interp, 2 = Cubic Interp
        DS_Rate = int(sys.argv[end_input_index+2])
        DS_Delay = int(sys.argv[end_input_index+3])
    # end of DS Segment to eventually add
    if False: #Potentially add Smoothing back into file
        SMOOTHING = int(sys.argv[end_input_index+4])
    # end of Smoothing Eventual Expansion

    if DATABASE_FLAG == 1:
        DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_OneHandOreba.pkl"
    elif DATABASE_FLAG == 2:
        DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_TwoHandOreba.pkl"
    elif DATABASE_FLAG == 3:
        DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_Clemson.pkl"
    elif DATABASE_FLAG == 4:
        DATABASE_FILEPATH = "./../Pickle_Databases/Dom_OneHandOreba.pkl"
    elif DATABASE_FLAG == 5:
        DATABASE_FILEPATH = "./../Pickle_Databases/Dom_Clemson.pkl"
    # elif DATABASE_FLAG == 6:
        # DATABASE_FILEPATH = "./../Pickle_Databases/BonusClemson.pkl"
    else:
        print("!!! WARNING: INVALID DATABASE SELECTION FLAG VALUE OF {} !!!".format(DATABASE_FLAG))
        print("Expects value of:")
        print("\t1 = OneHand OREBA")
        print("\t2 = TwoHand OREBA")
        print("\t3 = ClemCafe Data")
        print("\t4 = Clean ClemCafe Data")
        print("\t5 = Clean ClemCafe Data (Dom Hand GT ONLY)")
        print("\t6 = OneHand OREBA (Dom Hand GT ONLY)")
        print("Defaulting to use of ClemCafeData...")
        DATABASE_FILEPATH = "./../Pickle_Databases/ClemCafe.pkl"
    #end of Database switch statement



    SMOOTHING_FACTOR = 0 # the number of data points to either side to smooth the raw data
    WINDOWS_PER_BATCH = 128 # Batch-size for the model


    if MODEL_ARCH_FLAG == 4211 or MODEL_ARCH_FLAG == 4212: # if using hard coded input for TCN
        print("Hardcoding values for TCN Model!")
        if RESAMPLE_FLAG_FREQ == 0:
            if DATABASE_FLAG == 3 or DATABASE_FLAG == 5: # if Clemson
                DataFreq = 15
            else: # if OREBA
                DataFreq = 64
            # end switch Database
        else:
            DataFreq = RESAMPLE_FLAG_FREQ
        # end of if RESAMPLE_FLAG_FREQ == 0
        
        if MODEL_ARCH_FLAG == 4211:
            CUT_sec = 1023.0 / DataFreq # hardcode 1023 sequence length 9 TCN Depth
        elif MODEL_ARCH_FLAG == 4212: # Using 15 second hardcoded input for TCN
            CUT_sec = 255.0 / DataFreq # hardcode 255 sequence length 7 TCN Depth
        # end of switch MODEL_ARCH_FLAG
    # end if Hardcoding Values
        

    print('ModelArch Input is ',MODEL_ARCH_FLAG)
    print('CUT_sec Input is ',CUT_sec)
    print('STRIDE_sec input is ',STRIDE_sec)
    print('Training Data coming from file ',DATABASE_FILEPATH)
    print('RR_Flag Input is ',RR_FLAG)
    print('Fold_Index Input is ',FOLD_INDEX)
    print('Folds_Total Input is ',FOLDS_TOTAL)
    print('Resample Data down to Rate of {}'.format(RESAMPLE_FLAG_FREQ))


    print("python setup complete")


    print("Parsing Training Data...")
    start=time.time()

    if DATABASE_FLAG == 1 or DATABASE_FLAG == 2 or DATABASE_FLAG == 4: # if OREBA
        if RESAMPLE_FLAG_FREQ == 0:
            DataFreq = 64 # Frequency of Data Collection in [Hz]
        else:
            DataFreq = RESAMPLE_FLAG_FREQ # New [Hz] to which to convert signal
        # end if

        # Update Stride and Cut to Data Counts instead of seconds
        CUT = int(max(1,np.round(CUT_sec*DataFreq)))
        STRIDE = int(max(1,np.round(STRIDE_sec*DataFreq)))

        # Generate Data
        training_data,classes = GenerateDataAndClasses_OREBA_WinCnt(
                CUT, STRIDE,
                DATABASE_FILEPATH,
                FOLD_INDEX,	FOLDS_TOTAL, FOLD_SPLIT=RR_FLAG,
                RESAMPLE_FLAG=RESAMPLE_FLAG_FREQ,
                LIMIT_GT_HAND_FLAG = 0 # 0 by default, 1 = Limit to dominant hand only
                )
    elif DATABASE_FLAG == 3 or DATABASE_FLAG == 5: # if Clemson
        if RESAMPLE_FLAG_FREQ == 0:
            DataFreq = 15 # Frequency of Data Collection in [Hz]
        else:
            DataFreq = RESAMPLE_FLAG_FREQ # New [Hz] to which to convert signal
        # end if

        #Update Stride and Cut to Data Counts instead of seconds
        CUT = int(max(1,np.round(CUT_sec*DataFreq)))
        STRIDE = int(max(1,np.round(STRIDE_sec*DataFreq)))

        # Generate Data
        training_data,classes = GenerateDataAndClasses_ClemCafe_WinCnt(
                CUT,STRIDE,
                DATABASE_FILEPATH,
                FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=RR_FLAG,
                RESAMPLE_FLAG = RESAMPLE_FLAG_FREQ,
                #, RESAMPLE_FLAG=0, SMOOTHING=0 # optional flags not currently used for Paper Experiment
                LIMIT_GT_HAND_FLAG=0 # 0 by default, 1 = Limit to dominant hand only
                )
    else:
        print("ERROR: TRAINING DATA NOT SELECTED FOR DATABASEFLAG={}".format(DATABASE_FLAG))
        raise("NO TRAINING DATA PARSER SELECTED")
    #end of Database switch statement

    end=time.time()
    print("...Data parsed in ",end-start," seconds")
    
    
    
    # 1. Create your data matrices (as PyTorch tensors)
    # Assume X is your feature matrix and y is your label vector
    training_data = np.array(training_data).astype(np.float32) # convert to f32 for pytorch
    X_data = torch.from_numpy(training_data)
    X_data = torch.transpose(X_data, 1, 2) # flip axis to be [sample, channel, length]
    y_data = torch.from_numpy(classes.astype(np.float32))


    num_samples,total_axes,sample_length = np.shape(X_data)

    valid_idx = int(num_samples*validation_split)

    # 2. Create the TensorDataset
    # dataset = torch.utils.data.TensorDataset(X_data, y_data)

    # 3. Use a DataLoader to iterate over the dataset in batches
    # This handles batching, shuffling, and parallel loading automatically
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_dataset = torch.utils.data.TensorDataset(X_data[:-valid_idx], y_data[:-valid_idx])
    valid_dataset = torch.utils.data.TensorDataset(X_data[-valid_idx:], y_data[-valid_idx:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    
    # save a test batch for later testing
    sample_gen = iter(train_loader)
    sample_data, sample_trg = next(sample_gen)


    print("Finished Splitting Data.")





    ## import sys
    ## sys.exit()

    #######################################
    ###     CONSTANTS  FOR  MODELS      ###
    #######################################

    model, NUM_EPOCHS = ModelSelect_Torch(MODEL_ARCH_FLAG, sample_length, total_axes)
    
    
    
    
    # Send model to Device
    model = model.to(device)
    # @title Display model and number of parameters 
    # Run one batch through the model to initialize the lazy linear layer
    # this is necessary to get accurate parameter counts
    with torch.no_grad():
        model(sample_data.to(device)) 
    # end if with()
    
    print(model) # summarize Model
    print("Number of parameters in TallyNet model:", get_n_params(model))
    
    print("training_data of size ",training_data.shape)
    print("Classes of size ",classes.shape)
    
    print("Training...", flush=True)
    start=time.time()
    
    
    train_and_test(model, save_name = MODEL_NAME, epochs=NUM_EPOCHS)
    
    
    end=time.time()
    print("...Classifier trained in ",end-start," seconds")

 
    # OLD TENSORFLOW CODE
    #     if False: #If Using Checkpoints
    #         checkpoint_filepath = MODEL_NAME+'_chkpt'
    #         print("Checkpoint files stored at {}".format(checkpoint_filepath))
    #         model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #             filepath=checkpoint_filepath,
    #             save_weights_only=True,
    #             monitor='val_loss',
    #             mode='min',
    #             save_best_only=True)
    #         # Model weights are saved at the end of every epoch, if it's the best seen
    #         # so far.
    #         model.fit(training_data, classes, epochs=NUM_EPOCHS,
    #                         validation_split=0.05, verbose=2,
    #                         callbacks=[model_checkpoint_callback])

    #         # The model weights (that are considered the best) are loaded into the
    #         # model.
    #         model.load_weights(checkpoint_filepath)
    #     if True: # If Using Early Stopping
    #         PatienceCnt = 8
    #         print("Stopping Training when val_loss does not improve for {:d} Epochs".format(PatienceCnt))
    #         model_earlyStop_callback = keras.callbacks.EarlyStopping(
    #             monitor="val_loss",
    #             patience=PatienceCnt,
    #             restore_best_weights=True)
    #         # Model weights are saved at the end of every epoch, if it's the best seen
    #         # so far.
    #         model.fit(training_data, classes, epochs=NUM_EPOCHS,
    #                         validation_split=0.10, verbose=2,
    #                         callbacks=[model_earlyStop_callback])
    #     else:
    #         metrics = model.fit(training_data, classes, epochs=NUM_EPOCHS,
    #                         validation_split=0.05, verbose=2)
    #     # End of If Using Checkpoints

    #     end=time.time()
    #     print("...Classifier trained in ",end-start," seconds")

    #     # save model
    #     model.save(MODEL_NAME)



    # Test a small batch of values
    max_num_samples = min(40, len(training_data))
    data_sample = X_data[0:max_num_samples]
    class_sample = y_data[0:max_num_samples]
    print("Testing")

    print("Data sample has input of shape:")
    print(data_sample.shape)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        data_sample = data_sample.to(device)
        class_sample = class_sample.to(device)
        output = model(data_sample)
        output = output.squeeze(dim=1)
        test_loss += F.mse_loss(output, class_sample, reduction='sum').item()  # sum up batch loss


    test_loss /= len(valid_loader.dataset)

    print('Test loss (MSE):', test_loss )

    print('Actual, prediction:')
    #for a in range(0,len(classes)):
    for a in range(0,len(output)):
        print(class_sample[a],output[a])


