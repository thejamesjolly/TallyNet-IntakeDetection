'''
Filename: train-model.py
Original Code: Dr. Adam Hoover (train-model-5.py)
Edited By: James Jolly
'''



# removed InputLayer and using input_shape() in first layer instead to
# 	avoid hd5 model saving/loading bug
#
# 5-fold cross validation (out of 518 files, train on 414)
# testing is done using bc.c (C verson of classifier)
#


import logging
logging.getLogger('tensorflow').disabled = True

# TensorFlow, keras, np
import tensorflow as tf
from tensorflow import keras
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

from ModelSelectFuncs import ModelSelect




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

	# Define Variables

	num_samples,sample_length,total_axes = np.shape(training_data)





	## import sys
	## sys.exit()

	#######################################
	###     CONSTANTS  FOR  MODELS      ###
	#######################################

	model, NUM_EPOCHS = ModelSelect(MODEL_ARCH_FLAG, sample_length, total_axes)


	#end of Model Architecture Switch Statement

	model.compile(optimizer='adam',
				  loss='mean_squared_error',
				  metrics=['mean_absolute_error'])


	model.summary()

	print("training_data of size ",training_data.shape)
	print("Classes of size ",classes.shape)

	print("Training...")
	start=time.time()

	if False: #If Using Checkpoints
		checkpoint_filepath = MODEL_NAME+'_chkpt'
		print("Checkpoint files stored at {}".format(checkpoint_filepath))
		model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
			filepath=checkpoint_filepath,
			save_weights_only=True,
			monitor='val_loss',
			mode='min',
			save_best_only=True)
		# Model weights are saved at the end of every epoch, if it's the best seen
		# so far.
		model.fit(training_data, classes, epochs=NUM_EPOCHS,
						validation_split=0.05, verbose=2,
						callbacks=[model_checkpoint_callback])

		# The model weights (that are considered the best) are loaded into the
		# model.
		model.load_weights(checkpoint_filepath)
	if True: # If Using Early Stopping
		PatienceCnt = 8
		print("Stopping Training when val_loss does not improve for {:d} Epochs".format(PatienceCnt))
		model_earlyStop_callback = keras.callbacks.EarlyStopping(
			monitor="val_loss",
			patience=PatienceCnt,
			restore_best_weights=True)
		# Model weights are saved at the end of every epoch, if it's the best seen
		# so far.
		model.fit(training_data, classes, epochs=NUM_EPOCHS,
						validation_split=0.10, verbose=2,
						callbacks=[model_earlyStop_callback])
	else:
		metrics = model.fit(training_data, classes, epochs=NUM_EPOCHS,
						validation_split=0.05, verbose=2)
	# End of If Using Checkpoints

	end=time.time()
	print("...Classifier trained in ",end-start," seconds")

	# save model
	model.save(MODEL_NAME)




	max_num_samples = min(40, len(training_data))
	data_sample = np.array(training_data[0:max_num_samples])
	class_sample = np.array(classes[0:max_num_samples])
	print("Testing")
	test_loss, test_acc = model.evaluate(data_sample, class_sample)
	print('Test accuracy:', test_acc)



	print("Data sample has input of shape:")
	print(data_sample.shape)

	predictions = model.predict(data_sample)
	print('Actual, prediction:')
	#for a in range(0,len(classes)):
	for a in range(0,max_num_samples):
		print(classes[a],predictions[a])


