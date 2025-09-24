import logging
logging.getLogger('tensorflow').disabled = True

# TensorFlow, keras, np
import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
import time
import sys

from io import UnsupportedOperation
import pickle as pkl

# Add shared location for auxillary functions
sys.path.insert(1, './../AuxillaryFunctions')


# import personal Functions
from GenerateClassDataFuncs import GenerateDataAndClasses_ClemCafe_Bite
from GenerateClassDataFuncs import GenerateDataAndClasses_OREBA_Bite


print("Finished importing libraries and functions.")




if __name__ == '__main__':

	############################
	##### DEFINE CONSTANTS #####
	############################
	
	MODEL_NAME = sys.argv[1] # String folder to store data
	FOLDS_TOTAL = int(sys.argv[3])
	FOLD_INDEX = int(sys.argv[2])%FOLDS_TOTAL # wrap data into valid fold number in case fold 
					# index is given as [1,FOLDS_TOTAL] instead of [0,FOLDS_TOTAL) 
	DATABASE_FLAG = int(sys.argv[4]) # Database from which to pull the training and eval data
		# 1 = OHO (Dominant Motion)
		# 2 = THO (Both Hands)
		# 3 = ClemCafe Data
		# 4 = Clean ClemCafe
		# 5 = Clean ClemCafe (Dom GT Only)
		# 6 = OHO (Dominant GT Only)
	
	if False: # HardCoded Example
		MODEL_NAME = 'Example_HeydCNN_ClemSlurm/models/fold3'
		FOLD_INDEX = 3 # Which index to leave out for validation
		FOLDS_TOTAL = 5 # total number of folds to split the data into
		DATABASE_FLAG = 3 # Must be 1 (Dom Hand Only) or 2 (Both Hands) or 3(ClemCafe Data)
	# end of Hardcoded Example 

	CUT = 128 # 2 seconds @ 64 Hz 



	RR_FLAG = 1 # 0 = Block Fold Segmentation, 1 = Striped/Round Robin segmentation



	# 	# DOWN SAMPLE DATA INPUTS
	DOWN_SAMPLE_FLAG = 0 # 1 = Down Sample data, 0 = use raw data
	DOWN_SAMPLE_RATE = 1 # Num of points to consolidate into a single point
	DOWN_SAMPLE_OFFSET = 0 # Offset in DS data; in range [0, DS_Rate) 

	# NUMBER OF HANDS
	if DATABASE_FLAG == 1:
		DATABASE_FILEPATH = './../Pickle_Databases/Dnd_OneHandOreba.pkl'
		total_axes=6
		STRIDE = 32 # number of points to jump @ 64 Hz
	elif DATABASE_FLAG == 2:
		DATABASE_FILEPATH = './../Pickle_Databases/Dnd_TwoHandOreba.pkl'
		total_axes=12
		STRIDE = 32 # number of points to jump @ 64 Hz
	elif DATABASE_FLAG == 3:
		DATABASE_FILEPATH = './../Pickle_Databases/Dnd_Clemson.pkl'
		total_axes=6
		STRIDE = 96 # number of points to jump @ 15 Hz; Jumping forward by 3 seconds each to limit training examples due to large size of data
	elif DATABASE_FLAG == 4:
		DATABASE_FILEPATH = './../Pickle_Databases/Dom_OneHandOreba.pkl'
		total_axes=6
		STRIDE = 32 # number of points to jump @ 64 Hz
	elif DATABASE_FLAG == 5:
		DATABASE_FILEPATH = './../Pickle_Databases/Dom_Clemson.pkl'
		total_axes=6
		STRIDE = 96 # number of points to jump @ 15 Hz; Jumping forward by 3 seconds each to limit training examples due to large size of data
	### Removed Dataset using ~12 additional meals from Clemson Dataset which had Point GT labels but not gesture labels
    # elif DATABASE_FLAG == 6:
	# 	DATABASE_FILEPATH = './../Pickle_Databases/ClemCafe.pkl'
	# 	total_axes=6
	# 	STRIDE = 96 # number of points to jump @ 15 Hz; Jumping forward by 3 seconds each to limit training examples due to large size of data
	else:
		print("INVALID DATABASE SELECTION FLAG VALUE OF {}".format(DATABASE_FLAG))
		exit(0)
	# end of Database_Flag

	print("Finished Defining Constants.")





	#########################
	##### KNOBS TO TURN #####
	#########################

	### Knobs to Turn ###
	NUM_EPOCHS = 8 # Appears to stabilize after 3, so dropping from 30 to 5 since model takes a while to train
	WINDOWS_PER_BATCH = 256 # the number of samples to pull from files each time 
					# Value should maintain inequality [STRIDE*WINDOWS_PER_BATCH < 4.5M * (#folds-1)/#folds]
	USING_CLASS_WEIGHTS_FLAG=0
	if USING_CLASS_WEIGHTS_FLAG==1:
		# Weighting to be used for classes; 1 is undersampled and should have a higher weight
		Class_Weights_OREBA = {0: 1.0, 1: 4.0}

	# Define Variables 
	sample_length = CUT
	num_samples=WINDOWS_PER_BATCH

	print('CUT Input is ',CUT)
	print('STRIDE input is ',STRIDE)
	print('Training Data coming from file ',DATABASE_FILEPATH)
	print('FOLD_INDEX input is ',FOLD_INDEX)
	print('FOLDS_TOTAL input is ',FOLDS_TOTAL)

	print("\nPython setup complete")




	#######################################
	##### GENERATE MODEL ARCHITECTURE #####
	#######################################

	model = keras.Sequential()
	### FIRST PHASE: MICROMOVEMENT GESTURE PROBABILITIES ###

	#input as Convolution
	model.add(keras.layers.Conv1D(input_shape=(sample_length,total_axes,),
			filters=128,kernel_size=1,
			#strides=15,
			padding='valid',
			activation='relu'))

	# Additional Convolution Layers
	model.add(keras.layers.Conv1D(filters=128,kernel_size=3,
			padding='valid',
			activation='relu'))
	model.add(keras.layers.Conv1D(filters=128,kernel_size=5,
			padding='valid',
			activation='relu'))
	model.add(keras.layers.Conv1D(filters=128,kernel_size=7,
			padding='valid',
			activation='relu'))


	### SECOND PHASE: TIME MEMORY OF GESTURES ###
	model.add(keras.layers.LSTM(64, return_sequences=True,
			activation="tanh",recurrent_activation="hard_sigmoid"))

	model.add(keras.layers.LSTM(64, 
			activation="tanh",recurrent_activation="hard_sigmoid"))


	model.add(keras.layers.Flatten())  # must flatten to feed dense layer
	model.add(keras.layers.Dense(1))



	model.compile(optimizer='adam',
				  loss='mean_squared_error',
				  metrics=['mean_absolute_error'])
	#model.compile(optimizer='adam',
	#			  loss='binary_crossentropy',
	#			  metrics=['binary_crossentropy'])

	model.summary()






	#######################################
	##### GENERATE MODEL ARCHITECTURE #####
	#######################################


	# Create Training Data and Classes
	if DATABASE_FLAG == 1 or DATABASE_FLAG == 2 or DATABASE_FLAG == 4: # if using OREBA data at 64 Hz
		training_data,classes=GenerateDataAndClasses_OREBA_Bite(
			CUT, STRIDE, DATABASE_FILEPATH, 
			FOLD_INDEX, FOLDS_TOTAL, 
			FOLD_SPLIT = RR_FLAG, 
			DS_FLAG = DOWN_SAMPLE_FLAG,
			DS_Rate = DOWN_SAMPLE_RATE,
			DS_Delay = DOWN_SAMPLE_OFFSET
		)
	elif DATABASE_FLAG == 3 or DATABASE_FLAG == 5: # If Using ClemCafe Data at 15 Hz
		training_data,classes=GenerateDataAndClasses_ClemCafe_Bite(
			CUT, STRIDE, DATABASE_FILEPATH, 
			FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=RR_FLAG, RESAMPLE_FLAG=64
		)
	else: 
		print("INVALID SELECTION OF DATABASE_FLAG")
	# end of database switch 

	print("training_data of size ",training_data.shape)
	print("Classes of size ",classes.shape)
	print("Total Positive Class Examples: ",sum(classes))


	print("\nFinished Generating Training Classes and Targets.")




	#######################
	##### TRAIN MODEL #####
	#######################


	print("Training")
	start=time.time()

	if USING_CLASS_WEIGHTS_FLAG==0:
		metrics = model.fit(training_data, classes, epochs=NUM_EPOCHS,
					validation_split=0.15, verbose=2)
	elif USING_CLASS_WEIGHTS_FLAG==1:
		print("Using Class Weights for Training...")
		print(Class_Weights_OREBA)
		metrics = model.fit(training_data, classes, epochs=NUM_EPOCHS,
							class_weight=Class_Weights_OREBA,					
							validation_split=0.15, verbose=2)
	# end of if class weights

	end=time.time()
	print("classifier training",end-start," seconds")





	#######################################
	##### SAVE MODEL AND FINAL CHECKS #####
	#######################################

	# save model
	#model.save((sys.argv[1]+'fold'+sys.argv[2]+'of'+sys.argv[3]+'.h5'))
	#save model with lots of variable data in name

	#model.save((sys.argv[1]+'.h5')) #concise model name
	model.save(MODEL_NAME) #concise model name

	print("Testing")


	max_num_samples = min(40, len(training_data))
	data_sample=training_data[0:max_num_samples]
	classes_sample=classes[0:max_num_samples]

	test_loss, test_acc = model.evaluate(data_sample, classes_sample)
	print('Test accuracy:', test_acc)



	print("Data Has input of shape:")
	print(data_sample.shape)

	predictions = model.predict(data_sample)
	print('Actual, prediction:')
	#for a in range(0,len(classes)):
	for a in range(0,max_num_samples):
		print(classes_sample[a],predictions[a])
	# end of for samples

	print("\nFinished Running Training Script!")

# end of main()    













