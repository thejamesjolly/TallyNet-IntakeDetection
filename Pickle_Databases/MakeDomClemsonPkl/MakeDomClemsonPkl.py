'''
File: MakeDndClemsonPkl.py
Author: James Jolly
Purpose: 
	Create a pickle file of the Clemson Cafeteria dataset 
	using only Dominant ground truth gestures as ground truth events.
'''




####################
##     Imports    ##
####################


import sys
from io import UnsupportedOperation
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# Add shared location for auxillary functions
sys.path.insert(1, './../../AuxillaryFunctions')

from EvaluationMethodFuncs import TimePoint2Window
from EvaluationMethodFuncs import Window2TimePoint



#####################
##   Definitions   ##
#####################


def Normalize_Global_ZScore(WindowData, NO_GRAVITY_FLAG):
	'''
	Name: Normalize_Global_ZScore()
	Inputs: WindowData: 1D Array of data from a window
		consisting of some multiple of 6 measurements, lists in order of
			[x_accel, y_accel, z_accel, yaw, pitch, roll]
		NO_GRAVITY_FLAG: 0 if there is gravity in the accelerometer data, 1 if gravity has been removed
	Outputs: 1D Array of data adjusted to the means and StdDev defined from a
		previous parsing of all data (Run in Jan 2021)
	'''
	# Values computed by iterating over all data in the Cafeteria Dataset of Adam Hoover's website
	if NO_GRAVITY_FLAG == 0:
		MeansGlobal = [5.72467909e-01, -2.89774360e-01,  6.06284466e-01,  1.11284988e-15,  1.54272405e-15, 1.45554987e-16]
		StdDevGlobal = np.array([0.43334627,  0.32200813,  0.38929634, 18.13423163, 18.91109691, 37.66436347])
	elif NO_GRAVITY_FLAG == 1:
		# INVALID NUMBERS... ONLY USE GRAVITY DATA FOR CLEMSON DATA PICKLE
		print("Clemson Dataset only has gravity data... Using gravity normalization values...") 
		MeansGlobal = [5.72467909e-01, -2.89774360e-01,  6.06284466e-01,  1.11284988e-15,  1.54272405e-15, 1.45554987e-16] # USING GRAVITY NUMBERS
		StdDevGlobal = np.array([0.43334627,  0.32200813,  0.38929634, 18.13423163, 18.91109691, 37.66436347]) # USING GRAVITY NUMBERS


	# Save original shape for return
	OrigShape = WindowData.shape

	# Reshape for easy indexing
	WindowDataMat = np.reshape(WindowData,[-1,6])


	NormMat = np.zeros(np.shape(WindowDataMat)) #Preallocate space
	for b in range(0,6):
		NormMat[:,b] = (WindowDataMat[:,b]-MeansGlobal[b])/StdDevGlobal[b] # Zscore=(value-mean)/StdDev

	#convert back into a Original Shape matrix
	NormVector = np.reshape(NormMat,OrigShape)

	return(NormVector)
# end of def Normalize_ZScore()



#####################
##   Main Func()   ##
#####################

if __name__ == '__main__':
	
	##### FILE PATHS TO DATASET #####
	PROCESSED_DATA_FILEPATHS='dom_cleanCafe_data_files.txt'
	INTAKE_GESTURES_FILEPATHS='dom_cleanCafe_gesture_files.txt'


	##### UNUSED CONSTANTS #####
	# MAX_BITES_IN_WINDOW = 6
	# MAX_DATA = 54000  # one hour at 15 Hz
	# MAX_BITES = 10000  # maximum number of bites
	# MAX_WINDOWS = 20000
	# DATA_FIELDS = 7

	##### CONSTANTS #####
	SMOOTHING_FLAG = 0

	# Initialize empty lists
	AllData=[]
	UniqueID=[]
	DomHand=[]
	AllDataTimes=[]
	handednessList=[]

	with open("HandednessDemographics.txt",'r') as currFile:
		for line in currFile:
			handednessList.append(line.split('\t'))
			handednessList[-1][1]=handednessList[-1][1][:-1]
		# end for line
	# end with open()

	AllDataFiles = []
	with open(PROCESSED_DATA_FILEPATHS) as fpt_files:
		for filename in fpt_files:
			noNewLineName = (filename.splitlines())[0]
			AllDataFiles.append(noNewLineName)
		# end for line
	# end with open()



	###########################
	##### Dataset Parsing #####
	###########################

	cntr=0 #counter for user feedback
	for currFileName in AllDataFiles:

		# Compute Unique Descriptor based on File Name in Directories
		dirsToFile=currFileName.split('/')
		participantNum = dirsToFile[-3]
		courseNum = dirsToFile[-2]
		FileID=participantNum+courseNum

		# Find Handedness of participant
		MatchFoundFlag = 0
		for i in range(len(handednessList)):
			if participantNum==handednessList[i][0]:
				MatchFoundFlag = 1
				if handednessList[i][1]=='Right':
					DomHand.append(0)
				elif handednessList[i][1]=='Left':
					DomHand.append(1)
				else:
					print("ERROR IDENTIFYING DOMINANT HAND")
					DomHand.append(-1)
				break
		if MatchFoundFlag == 0:
			print("ERROR FINDING PARTICIPANT NUMBER IN HAND DEMOGRAPHIC FILE")
			print("UNABLE TO FIND PARTICIPANT NUMBER {}".format(participantNum))
			DomHand.append(-1)



		# /* read data file, determine total amount of data */
		totalData = 0
		# /* file format is x y z (accel units are volts)
		# ** yaw pitch roll (gyro units are volts) scale (units are grams) */
		zero = np.zeros(shape=(3, 1))

		fpt = open(currFileName, 'r')
		Data=np.array([[float(x) for x in line.split()] for line in fpt])
		fpt.close()

		Data = Data[:,0:6] # Remove Scaling data column
		Data = np.transpose(Data)
		for j in range(0, 3):
			zero[j] = np.sum(Data[j+3][:])
		totalData = len(Data[0])


		for j in range(0, 3):
			zero[j] /= totalData



		# /* convert data voltages to deg/sec (gyros)
		# ** and gravities (accelerometers)
		# ** gyro=LPY410al, 2.5mv per deg/sec, zero-point found
		# ** by calculating the average data value of the whole recording
		# ** accel=LIS344alh, Vdd=3.3v, 5/3.3=1.515 gravities per volt */
		for i in range(0, totalData):
			for j in range(0, 3):
				Data[j][i] = (Data[j][i]-1.65)*(5.0/3.3)
			for j in range(3, 6):
				Data[j][i]=(Data[j][i]-zero[j-3])*400.0

		Data = np.transpose(Data)

		if SMOOTHING_FLAG==1:
			# /* smooth the data */
			# Copy data in front and behind smoothing window
			for i in range(0, SMOOTHING): 
				for j in range(0, DATA_FIELDS):
					SmoothedData[j][i] = Data[j][i]
			for i in range(totalData-SMOOTHING, totalData):
				for j in range(0, DATA_FIELDS):
					SmoothedData[j][i] = Data[j][i]


			# averaging over a window centered on datum
			for i in range(SMOOTHING, totalData - SMOOTHING):
				for j in range(0, DATA_FIELDS):
					total = 0.0
					for k in range(i-SMOOTHING, i+SMOOTHING+1):
						if (k >= 0 and k < totalData):
							total += Data[j][k]
					SmoothedData[j][i] = total / (SMOOTHING*2 + 1)
		# end of if SMOOTHING statement


		# Standardize Data with Global Z-Score
		norm_Data = Normalize_Global_ZScore(Data,0) # Pass zero as flag since the accelerometer still has gravity 
		# norm_Data=Data



		# Convert indexes to zero based time at 15Hz
		data_times=list(range(0,totalData))
		data_times=(np.array(data_times))/15.0
		#for i in range(totalData):
		#	data_times[i]=float(data_times[i])/15.0	
		UniqueID.append(FileID)
		AllData.append(np.array(norm_Data))
		AllDataTimes.append(np.array(data_times))

		# Update user on progress through meals
		#print("Completed Meal {}".format(currFileName)) 
		cntr+=1
		if cntr%15==0:
			print("Through File {} of {}".format(cntr,len(AllDataFiles)))
	# end of for currFileName loop

	print("Finished With Meals")




	##########################
	##### GT Annotations #####
	##########################

	AllIntakeGestureFiles = []
	with open(INTAKE_GESTURES_FILEPATHS) as fpt_files:
		for filename in fpt_files:
			noNewLineName = (filename.splitlines())[0]
			AllIntakeGestureFiles.append(noNewLineName)
		# end of for
	# end of open()

	AllBites = [] # initialize empty list
	# AllBites_Handedness = [] # All Bites are with the dominant hand
	AllGestures = [] # initialize empty list
	AllGestures_Durations = [] # durations


	for FileIdx in range(len(AllIntakeGestureFiles)):

		# //load bites.txt
		currFileName = AllIntakeGestureFiles[FileIdx]
		try:
			with open(currFileName, 'r') as fpt:
				GT_Window_Data = [line.split() for line in fpt]
			# End of fpt; close file
		except:
			print("skipping Meal {}...".format(currFileName))
			continue
		# End of try file read

		totalGestures = len(GT_Window_Data)
		GTbiteWindows = [] # clear from previous file
		Durations=[]

		for a in range(0,totalGestures):
			if (GT_Window_Data[a][0].lower() == "bite") or (GT_Window_Data[a][0].lower() == "drink"):
				# Add pair of [start, end] data point
				GTbiteWindows.append([int(GT_Window_Data[a][1]), int(GT_Window_Data[a][2])])
				Durations.append(float(int(GT_Window_Data[a][2])-int(GT_Window_Data[a][1]))/15.0)
			#end if
		#end of for totalGestures loop

		BiteTimes = Window2TimePoint(np.array(GTbiteWindows))



		### Once Complete processing, Add all meal annotations to the Pickle variables
		AllGestures.append(np.array(GTbiteWindows))
		AllGestures_Durations.append(Durations)

		AllBites.append(BiteTimes)



		if FileIdx % 50 == 0:
			print("Through {} of {} files...".format(FileIdx,len(AllIntakeGestureFiles)))
	# end of for currFileName loop


	print("Finished with Gesture Annotations")


	#########################
	##### Output Pickle #####
	#########################

	# Define Values Used in Normalization
	MeansGlobal_rawAcc = np.array([5.67246264e-01, -2.89351987e-01, 6.06367364e-01, 1.07032646e-15, 2.00383723e-15, -3.91199091e-16]) # Data contains Gravity
	StdDevGlobal_rawAcc = np.array([0.43807806, 0.32371267, 0.38916464, 18.0970793, 18.85461884, 37.44610756]) # Data contains Gravity

	README = [["INTRODUCTION",
			   'This pkl dataset contains all data from the Clemson Cafeteria Dataset processed for easy use in Python in IMU event tracking.',
			   'GT Labels are taken from the gestures_union.txt files and the intake moments correspond to the middle of each gesture.  This is only the dominant (recorded) hand gesture targets.'
			  ],
			  ["DICTIONARY ENTRIES IN DATABASE",
			   'UniqueID: Contains a participant and course number to uniquely identify each meal file in the original Database.',
			   'proc_data_rawAcc: list of IMU data for each meal, processed to normalize per axis with Global Z-Score using the means and standard deviations for rawAcc.  each row formatted in [acc_1, acc_2, acc_3, yaw, pitch, roll].  Unconfirmed actual orientation of sensors, or if gyro data aligns to the order of accelerometers. No other processing has been done (e.g. gravity removal or smoothing).',
			   #'data_times: time [sec] from the start of the data, used to index answer.  Can be computed as index/15 since data is at 15Hz.',
			   'handedness: list with an entry for each meal of whether the user was left or right hand dominant, and thus on which hand the data collection device was.',
			   'bites_gt: list of numpy array containing the indices of each intake moment (the specific moment food entered the mouth). Can convert to time since recording started by simply dividing value by 15 (since data is at 15Hz).',
			   'bites_gt_window: list of numpy array containing the indices of each intake gesture (start and end of motion). Can convert to time since recording started by simply dividing value by 15 (since data is at 15Hz).',
			   # 'bites_handedness_gt: list of lists, containing whether the corresponding ground truth bite was eaten with the left, right, or both hands.',
			   'means_global_rawAcc: Means for each axis used to normalize the data in proc_data_rawAcc.',
			   'stddev_global_rawAcc: Standard Deviations for each axis used to normalize the data in proc_data_rawAcc.',
			  ]
			 ]

	ClemsonCafe_Cucumber={
		'README': README,
		'UniqueID': UniqueID,
		'proc_data_rawAcc': AllData,
		#'data_times': AllDataTimes, #dont include since the info is easy to compute as { data_times[index] = index/15 }
		'handedness': DomHand,
		'bites_gt': AllBites,
		'bite_gt_window': AllGestures,
		# 'bites_handedness_gt': AllBites_Handedness, #Not Needed in Dom Hand Pickle. used in Mixed Gesture target pickle.
		'means_global_rawAcc': MeansGlobal_rawAcc,
		'stddev_global_rawAcc': StdDevGlobal_rawAcc,
	}

	print("Finished Making Cucumber for Pickling")



	outfile = open("../Dom_Clemson.pkl",'wb')
	pkl.dump(ClemsonCafe_Cucumber,outfile)
	outfile.close()

	print("Finished Pickling ClemsonCafe Dataset")
	
	
# end of main()


'''
##### CALCULATE MEAN AND STD_DEV IF NEEDED FOR NORMALIZE DATA FUNCTION
##### THIS IS PREVIOUSLY CALCULATED AND HARDCODED TO SAVE TIME IN THE FUNCTION
##### COMMENTING OUT THE NORMALIZE FUNCTION AND THEN  RUNNING THIS SECTION SHOULD RECREATE THE VALUES USED. 


### Calculate Mean
Summations_RawMean=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
totalPts=0
for i in range(len(AllData)):
	totalPts+=len(AllData[i])
	for j in range(len(AllData[i])):
		Summations_RawMean+=AllData[i][j,:]
GlobalMeans_G = Summations_RawMean/float(totalPts)
print(GlobalMeans_G)


### Calculate StdDev
# GlobalMeans_G=[0.56744746,-0.28966115,0.60622525,-0.0008205,-0.00089252,-0.0005668]
Summations_RawVar=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for i in range(len(AllData)):
	for j in range(len(AllData[i])):
		Summations_RawVar+=np.power(AllData[i][j,:]-GlobalMeans_G,2)
GlobalVariance_G=Summations_RawVar/float(totalPts)
GlobalStdDev_G=np.power(GlobalVariance_G,0.5)
print(GlobalStdDev_G)


### PRINT STATEMENT OF VALUES 

print('\tGlobalMeans_G')
for i in range(len(GlobalMeans_G)):
	print('{:.6f}'.format(GlobalMeans_G[i]))
print()

print('\tGlobalStdDev_G')
for i in range(len(GlobalStdDev_G)):
	print('{:.6f}'.format(GlobalStdDev_G[i]))
print()


### FINAL NOTES OF CALCULATED VALUES

GlobalMeans_withG=[5.72467909e-01, -2.89774360e-01,  6.06284466e-01,  1.11284988e-15,  1.54272405e-15, 1.45554987e-16]
GlobalStdDev_withG=[0.43334627,  0.32200813,  0.38929634, 18.13423163, 18.91109691, 37.66436347]

'''