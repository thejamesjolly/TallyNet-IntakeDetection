'''
File: ModelSelectFuncs.py
Version: 3.0
Author: James Jolly
Purpose: 
	Allow for flexible Model Architecture layers inside TallyNet
Version History:
	3.0 - Huge sweep across layer parameters with architectures with number "1XYZ", 
			where X is conv filter count, y is LSTM node code, 
			and Z is the # of hidden nodes in dense layer 
	2.0 - Simplification of Models to best performing 101 and 118 from search
	1.0 - Intial creation of file
'''
# TensorFlow, keras, np
import tensorflow as tf
from tensorflow import keras

# Function: ModelSelect
# Purpose: Seperated function to have a separate space for model architectures to reduce the 
#          number of lines in the Training File (since many architectures are lenghty code) 
# Inputs: MODEL_ARCH_FLAG: switch value (int) that selects which model to use
# Outputs: model: the keras model used for training
#          NUM_EPOCHS: the number of epochs used to train this model (sometimes reduced for 
#                     longer architectures to save training time)

def ModelSelect(MODEL_ARCH_FLAG, sample_length, total_axes, ):

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
	