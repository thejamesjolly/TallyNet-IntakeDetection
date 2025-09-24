'''
File: GenerateClassDataFuncs.py
Written by: James Jolly
Purpose: House multiple functions to interface between different datasets (Clemson Cafeteria, Oreba 1Hand, and Oreba 2Hand)
		and different model Ground truths (bite/no bite versus a total count in the window).
TABLE of CONTENTS
	Resample Funcs
		LinearInterp_DS()
		CubicInterp_DS()
		LinearInterp()
		CubicInterp()
	Testing/Validation/Evaluation Split Funcs
		TE_Training_Split()
		TE_Eval_Split()
		# Possibly Add a TVE split for Testing, Validation, and Eval splits
	Database to Data&Class Functions
		GenerateDataAndClasses_OREBA_Bite()
		GenerateDataAndClasses_OREBA_WinCnt()
		GenerateDataAndClasses_ClemCafe_WinCnt()
		GenerateDataAndClasses_ClemCafe_Bite()
	Database to EvalData Functions
		GenerateEvalData_OREBA()
		GenerateEvalData_ClemCafe()
		
Segments
'''

import numpy as np
import scipy
from scipy import interpolate
import math

from io import UnsupportedOperation
import pickle as pkl



def LinearInterp_DS(Data, DS_Rate, DS_Delay):
	'''
	Func: LinearInterp_DS()
	Purpose: Convert raw data into a resampled version, which is downsampled and then linearly interpolated back up
		INPUTS
	Data     - Original data in a numpy array where time is the 0th axis
	DS_Rate  - The rate at which to downsample the data.  For instance, a 64Hz signal with DS_rate=4 will be 64/4=16Hz
	DS_Delay - An offset to allow the data to be translated across the dataset rather than always 
				anchoring on the first data point
		OUTPUT
	DS_data  - Same shape as the original data, with linear interpolated results 
	'''
	
	print("Down sampling meal by rate of {}".format(DS_Rate))
	print("Using offset of {}".format(DS_Delay))
	print("Using Linear interpolation...")
	
	DS_data=np.zeros(Data.shape)
	for i in range(DS_Rate,len(Data)-DS_Rate):
		if (i-DS_Delay)%DS_Rate==0:
			DS_data[(i),:]=Data[(i),:]
		else:
			last_data=Data[(i)-((i-DS_Delay)%DS_Rate),:]
			next_data=Data[(i)-((i-DS_Delay)%DS_Rate)+DS_Rate]
			DS_data[(i),:]=last_data+(next_data-last_data)*(((i-DS_Delay)%DS_Rate)/DS_Rate)
	
	# copy over data that could not be interpolated rather than copying zeros
	for i in range(0,DS_Rate):
		DS_data[(i),:]=Data[(i),:]
	for i in range(len(Data)-DS_Rate,len(Data)):
		DS_data[(i),:]=Data[(i),:]
	
	# Return Final Results
	return DS_data
# end of LinearInterp_DS()



def CubicInterp_DS(Data, DS_Rate, DS_Delay):
	'''
	Func: CubicInterp_DS()
	Purpose: Convert raw data into a resampled version, which is downsampled and then linearly interpolated back up
		INPUTS
	Data     - Original data in a numpy array where time is the 0th axis
	DS_Rate  - The rate at which to downsample the data.  For instance, a 64Hz signal with DS_rate=4 will be 64/4=16Hz
	DS_Delay - An offset to allow the data to be translated across the dataset rather than always 
				anchoring on the first data point
		OUTPUT
	DS_data  - Same shape as the original data, with linear interpolated results
	'''
	
	print("Down sampling meal by rate of {}".format(DS_Rate))
	print("Using offset of {}".format(DS_Delay))
	print("Using cubic interpolation...")
	
	
	[num_samples,num_sensor_axes] = np.shape(Data)
	
	
	y = np.array([Data[0,:]]) # Grab first item to initialize size, delete after grabing data
	for i in range(DSrate,len(Data)-DSrate):
		if (i-DS_Delay)%DSrate==0:
			y= np.append(y,[Data[(i),:]],axis=0)
	y = y[1:len(y),:] #delete intialization row
	x = list(range(DSrate+DS_Delay,len(Data)-DSrate, DSrate))

	xnew = list(range(DSrate+DS_Delay,len(Data)-2*DSrate))
	ynew=np.zeros([len(Data)-(3*DSrate + DS_Delay),num_sensor_axes])
	for i in range(num_sensor_axes):
		print("interpolating axis {}".format(i))
		f = scipy.interpolate.interp1d(x,y[:,i],'cubic')
		ynew_temp = f(xnew)
		ynew[:,i] = ynew_temp


	meal_data=np.zeros([DSrate+DS_Delay,num_sensor_axes])
	meal_data=np.append(meal_data,ynew,axis=0)
	meal_data=np.append(meal_data,np.zeros([2*DSrate,num_sensor_axes]),axis=0)
	
	
	# copy over data that could not be interpolated rather than copying zeros
	for i in range(0,DSrate+DS_Delay):
		meal_data[(i),:]=Data[(i),:]
	for i in range(len(meal_data-(2*DSrate)),len(meal_data)):
		meal_data[(i),:]=Data[(i),:]
	
	return meal_data

# end of CubicInterp_DS()


def LinearInterp(Data, OrigFreq, NewFreq):
	'''
	Func: LinearInterp()
	Purpose: Convert raw data into a resampled version, which is linearly interpolated from
				raw and then resampled up or down to desired frequency
		INPUTS
	Data     - Original data in a numpy array where time is the 0th axis
	OrigFreq - Original Frequency of the Data, given in [Hz]
	NewFreq  - Desired resampled Frequency of the Data, given in [Hz]
		OUTPUT
	DS_data  - Same shape as the original data, with linear interpolated results 
	'''
	
	print("Down sampling meal by rate of {}".format(DS_Rate))
	print("Using offset of {}".format(DS_Delay))
	print("Using Linear interpolation...")

	
	# Return Final Results
	return DS_data
# end of LinearInterp()



def ResampleData(Data, OrigFreq, NewFreq, InterpFlag=0):
	'''
	Func: CubicInterp()
	Purpose: Convert raw data into a resampled version, which is interpolated with a cubic
				function on full raw data and then resampled up or down to desired frequency
		INPUTS
	Data     - Original data in a numpy array where time is the 0th axis and independent
					sensor readings along the next axis
	OrigFreq - Original Frequency of the Data, given in [Hz]
	NewFreq  - Desired resampled Frequency of the Data, given in [Hz]
	InterpFlag - which interpolation to use
					0 = Default (cubic)
					1 = Linear
					2 = Cubic
					
		OUTPUT
	DS_data  - MOST of Data, but with slight clipping at the end to maintain valid interpolation range
				same format as data
	'''
	
	VERBOSE_FLAG = 0
	
	# Print out interpolation data for tracing after training
	
	if InterpFlag == 1:
		if VERBOSE_FLAG == 1:
			print("Using linear interpolation...")
		# end VERBOSE_FLAG
	elif InterpFlag == 2:
		if VERBOSE_FLAG == 1:
			print("Using cubic interpolation...")
		# end VERBOSE_FLAG
	else:
		if VERBOSE_FLAG == 1:
			if InterpFlag != 0: # print warning if invalid flag value
				print("!!! WARNING: INVALID VALUE  OF \"{}\" FOR INTERPOLATION SCHEME FLAG!!!".format(InterpFlag))
				print("Values must be: 0 = Default, 1 = Linear, 2 = Cubic")
				print("Switching to Default...")
			print("Using default interpolation, currently set to cubic...")
		# end VERBOSE_FLAG
		InterpFlag = 2
	# end of InterpFlag Prints 
		
	
	[num_samples,num_sensor_axes] = np.shape(Data)

	y = Data
	x = list(range(len(Data)))
	
	# create enough points to span most of the linear space of the data at the newly sampled frequency
	# NOTE: Include the "-1" in order to allow the interpolation to maintain at least one point past the 
	#          largest new value
	xnew = (np.arange(0,int(np.floor((len(Data)-1)*(float(NewFreq)/float(OrigFreq))))))
	xnew = xnew*(float(OrigFreq)/float(NewFreq)) # scale to corresponding locations
	ynew=np.zeros([len(xnew),num_sensor_axes])

	for i in range(num_sensor_axes):
		if VERBOSE_FLAG == 1:
			print("interpolating axis {}".format(i))
		# end VERBOSE_FLAG
		if InterpFlag == 1:
			f = scipy.interpolate.interp1d(x,y[:,i],'linear')
		elif InterpFlag == 2:
			f = scipy.interpolate.interp1d(x,y[:,i],'cubic')
		ynew_temp = f(xnew)
		ynew[:,i] = ynew_temp
	
	
	meal_data=np.array(ynew)
	# # # # # # WARNING # # # # # #
	# Data is clipped off the back of resampling.  This will only be a small amount based
	#   on the ratio of the original and new frequencies. Data should not be meaningful this 
	#   close to the end of the meal
	
	return meal_data

# end of ResampleData()






def TE_Training_Split(NumMeals, FOLD_SPLIT, FOLDS_TOTAL, FOLD_INDEX):
	'''
	Func:
	Purpose:
		INPUTS
	NumMeals - The total number of meals
	FOLD_SPLIT - either using block fold division (0) or striped division of files (1)
	FOLDS_TOTAL - Either the number of folds (2+), or not dropping any data (0 or 1); leave one subject out not implemented on OREBA
	FOLD_INDEX  - The fold number from [0,FOLDS_TOTAL)
		OUTPUTS
	meals_for_fold - a list of the indices of each meal to be used for Training 
	'''

	if FOLDS_TOTAL>1:
		if FOLD_SPLIT==0: # Initial Split of data into fifths 
			print("Training on Data Split into contiguous fifths...")
			meals_for_fold=[]
			start_exclusion=math.ceil(FOLD_INDEX*NumMeals/FOLDS_TOTAL) #first file to be exlcuded
			end_exlcusion=math.ceil((FOLD_INDEX+1)*NumMeals/FOLDS_TOTAL) #first file to be included after fold ends
			for i in range(NumMeals):
				if i < start_exclusion or i >= end_exlcusion:
					meals_for_fold.append(i)
		elif FOLD_SPLIT==1: # Split by round robin of every few files
			print("Training on Data Split by Round Robin approach...")
			meals_for_fold=[]
			meals_notTraining_fold=list(range(FOLD_INDEX,NumMeals,FOLDS_TOTAL))
			for i in range(NumMeals):
				ExcludeFlag=0 # clear flag
				for j in range(len(meals_notTraining_fold)): # Search all meals not to be included for i-th meal
					if i==meals_notTraining_fold[j]:
						ExcludeFlag=1
						break # if i-th meal found, break j loop
				#end of j loop
				if ExcludeFlag==0: # If not excluding
					meals_for_fold.append(i)
		else: # Incorrect value for segmentation option flag, default to block segmentation
			print("!!! INVALID FOLD_SPLIT FLAG VALUE.  SHOULD BE 0 (BLOCK) OR 1 (STRIPED) !!!")
			print("Defaulting to block data...")
			print("Training on Data Split into {} contiguous blocks...".format(FOLDS_TOTAL))
			meals_for_fold=[]
			start_exclusion=math.ceil(FOLD_INDEX*NumMeals/FOLDS_TOTAL) #first file to be exlcuded
			end_exlcusion=math.ceil((FOLD_INDEX+1)*NumMeals/FOLDS_TOTAL) #first file to be included after fold ends
			for i in range(NumMeals):
				if i < start_exclusion or i >= end_exlcusion:
					meals_for_fold.append(i)
	elif ((FOLDS_TOTAL == 1) or (FOLDS_TOTAL == 0)): # Use all of the meals
		meals_for_fold=range(NumMeals)
	
	return meals_for_fold

# end of TE_Training_Split()




def TE_Eval_Split(NumMeals, FOLD_SPLIT, FOLDS_TOTAL, FOLD_INDEX):
	'''
	Func:
	Purpose:
		INPUTS
	NumMeals - The total number of meals
	FOLD_SPLIT - either using block fold division (0) or striped division of files (1)
	FOLDS_TOTAL - Either the number of folds (2+), or not dropping any data (0 or 1); leave one subject out not implemented on OREBA
	FOLD_INDEX  - The fold number from [0,FOLDS_TOTAL)
		OUTPUTS
	meals_for_fold - a list of the indices of each meal to be used for Training 
	'''
	
	VERBOSE_PRINT = 0
	
	if FOLDS_TOTAL>1:
		if FOLD_SPLIT==0: # Initial Split of data into fifths 
			if VERBOSE_PRINT==1:
				print("Training on Data Split into contiguous fifths...")
			meals_for_fold=[]
			start_exclusion=math.ceil(FOLD_INDEX*NumMeals/FOLDS_TOTAL) #first file to be exlcuded
			end_exlcusion=math.ceil((FOLD_INDEX+1)*NumMeals/FOLDS_TOTAL) #first file to be included after fold ends
			for i in range(NumMeals):
				if i >= start_exclusion and i < end_exlcusion:
					meals_for_fold.append(i)
		elif FOLD_SPLIT==1: # Split by round robin of every few files
			if VERBOSE_PRINT==1:
				print("Training on Data Split by Round Robin approach...")
			meals_for_fold=list(range(FOLD_INDEX,NumMeals,FOLDS_TOTAL))
		else: # Incorrect value for segmentation option flag, default to block segmentation
			print("!!! INVALID FOLD_SPLIT FLAG VALUE.  SHOULD BE 0 (BLOCK) OR 1 (STRIPED) !!!")
			print("Defaulting to block data...")
			print("Training on Data Split into {} contiguous blocks...".format(FOLDS_TOTAL))
			meals_for_fold=[]
			start_exclusion=math.ceil(FOLD_INDEX*NumMeals/FOLDS_TOTAL) #first file to be exlcuded
			end_exlcusion=math.ceil((FOLD_INDEX+1)*NumMeals/FOLDS_TOTAL) #first file to be included after fold ends
			for i in range(NumMeals):
				if i >= start_exclusion and i < end_exlcusion:
					meals_for_fold.append(i)
	elif ((FOLDS_TOTAL == 1) or (FOLDS_TOTAL == 0)): # Use all of the meals
		meals_for_fold=range(NumMeals)
	
	return meals_for_fold

# end of TE_Eval_Split




def GenerateDataAndClasses_OREBA_Bite(CUT, STRIDE, DATABASE_FILEPATH, FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=1, DS_FLAG=0, DS_Rate=1, DS_Delay=0):
	'''
	Func: GenerateDataAndClasses_OREBA_Bite()
	Purpose: Connects OREBA Dataset (both one and two hand) to bite/no-bite classes
		Inputs:
	CUT - Total number of Datapoints in a window of data
	STRIDE  - number of data points to move between each generated window
	DATABASE_FILES_NAMES - file address of the Pickle compressed data file, containing either one-handed or two-handed data
	FOLD_INDEX  - The fold number from [0,FOLDS_TOTAL)
	FOLDS_TOTAL - Either the number of folds (2+), or not dropping any data (0 or 1); leave one subject out not implemented on OREBA
	FOLD_SPLIT - either using block fold division (0) or striped division of files (1)
	DS_FLAG - whether to downsample data (1=LinearInterp, 2=CubicInterp) or to use the original signal (0)
	DS_Rate - # of points to combine when downsampling, then linearly interpolate.  
				e.g. DS_Rate=4 means that the signal will drop from 64 --> 16 Hz by grabbing every 4th data point
	DS_Delay - offset used in the selection of the data, in range [0,DS_Rate)
		Outputs:
	np_training_data - windows of data that are CUT data wide and at least STRIDE data offset from each other
	np_classes - corresponding classes to each window in np_training_data, where 1=bite and 0=non-bite
	
	'''
	
	
	
	# Output Message based on Down Sample Flag
	if DS_FLAG==1:
		print("Down sampling meal by rate of {}".format(DS_Rate))
		print("Using offset of {}".format(DS_Delay))
		print("Using Linear Interpolation.")
	elif DS_FLAG==2:
		print("Down sampling meal by rate of {}".format(DS_Rate))
		print("Using offset of {}".format(DS_Delay))
		print("Using Cubic Interpolation.")
	
	
	with open(DATABASE_FILEPATH,'rb') as fh:
		dataset = pkl.load(fh)
	#OREBA_Cucumber={'UniqueID','proc_data','handedness','bites_gt'}
	OrigFreq = 64 # [Hz] Oreba data collectioni rate
	
	proc = dataset['proc_data']
	biteGT_times=dataset['bites_gt']
	totalMeals = len(proc)

	training_data=[]
	classes=[]
	RequiredOverlapPercent = 0.8 # percentage of cut window that needs to be within bite_gt
					# in order to be marked as a bite class

	
	# Split the data into training and testing data
	meals_for_fold = TE_Training_Split(totalMeals, FOLD_SPLIT, FOLDS_TOTAL, FOLD_INDEX)



	for meal_num in meals_for_fold:

		meal_times=dataset['data_times'][meal_num]

		OG_meal_data=proc[meal_num][:,:]
		######### DOWN SAMPLING SECTION ###########
		if DS_FLAG==1: # Linear Interpolate Points to down sample and upsample
			DS_data=LinearInterp(OG_meal_data,DS_Rate,DS_Delay)
			meal_data=DS_data
		elif DS_FLAG==2: # Cubic interpolation
			DS_data=CubicInterp(OG_meal_data,DS_Rate,DS_Delay)
			meal_data=DS_data
		######### End of Downsampling Section #########
		else: #No Downsampling
			meal_data=OG_meal_data


		currMealBiteTimes=biteGT_times[meal_num]

		WindowTimeDuration=meal_times[0+CUT]-meal_times[0]
		if True: # Original interpretation of Bites
			for time_rover in range(0,len(meal_times)-CUT,STRIDE):

				InBiteFlag=-1 #clear the flag; -1 is no match, non-negative value is the location of match, -2 is error
				PartBiteFlag=0 # set to 1 if the window is more than 20% overlapping with a bite
				for i in range(len(currMealBiteTimes)):
					# if ((leftmost end of window) - (rightmost start of window))>0, then windows are overlapping 
					overlap=min(meal_times[time_rover+CUT], currMealBiteTimes[i,1]) - max(meal_times[time_rover], currMealBiteTimes[i,0]) 
					if overlap > 0:
						# if at least required overlap, then mark as bite and gt match
						if overlap/WindowTimeDuration >= RequiredOverlapPercent:
							InBiteFlag=i
						# if greater than non-bite portion does not have required overlap percent
						elif overlap/WindowTimeDuration >= 1-RequiredOverlapPercent:
							PartBiteFlag=1
				# end of InBiteFlag check

				if InBiteFlag >= 0: # if a match was found, export positive class
					training_data.append(meal_data[time_rover:time_rover+CUT,:])
					classes.append(1)
				elif PartBiteFlag == 0: # If it didn't significantly overlap with a bite window
					training_data.append(meal_data[time_rover:time_rover+CUT,:])
					classes.append(0) # output negative class
				# else:
					# Do nothing since the window strattles clear data to avoid giving blurred examples to model

			# end of time_rover loop

		if False: #Use Robitic Bites of just learning the start of a window
			print("Unimplemented Bite Class Encoder...")

	#end of meal_num loop	


	np_classes=np.array(classes)
	np_training_data=np.array(training_data)

	return(np_training_data,np_classes)

# End of def GenerateFicDataAndClasses_OREBA()






def GenerateDataAndClasses_OREBA_WinCnt(CUT, STRIDE, DATABASE_FILEPATH, FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=1, DS_FLAG=0, DS_Rate=1, DS_Delay=0, RESAMPLE_FLAG=0, LIMIT_GT_HAND_FLAG=0):
	'''
	Func: GenerateDataAndClasses_OREBA_WinCnt()
	Purpose: Connects OREBA Dataset (both one and two hand) to window count classes
		INPUTS:
	CUT - Total number of Datapoints in a window of data
	STRIDE  - number of data points to move between each generated window
	DATABASE_FILES_NAMES - file address of the Pickle compressed data file, containing either one-handed or two-handed data
	FOLD_INDEX  - The fold number from [0,FOLDS_TOTAL)
	FOLDS_TOTAL - Either the number of folds (2+), or not dropping any data (0 or 1); leave one subject out not implemented on OREBA
	FOLD_SPLIT - either using block fold division (0) or striped division of files (1)
	DS_FLAG - whether to downsample data (1=LinearInterp, 2=CubicInterp) or to use the original signal (0)
	DS_Rate - # of points to combine when downsampling, then linearly interpolate.  
				e.g. DS_Rate=4 means that the signal will drop from 64 --> 16 Hz by grabbing every 4th data point
	DS_Delay - offset used in the selection of the data, in range [0,DS_Rate)
	RESAMPLE_FLAG - whether to change the data to a new frequency; 0 if not, new frequency in [Hz] otherwise
	LIMIT_GT_HAND_FLAG - Flag to limit the GT's targeted by the model;
					0 = All GT intakes (Both Hands), 1 = Dominant Hand Only, 2 = Non-Dominant Hand Only
		OUTPUTS:
	np_training_data - windows of data that are CUT data wide and at least STRIDE data offset from each other
	np_classes - corresponding classes to each window in np_training_data, which is the number of bites in the window
	'''
	
	# Output Message based on Down Sample Flag
	if DS_FLAG==1:
		print("Down sampling meal by rate of {}".format(DS_Rate))
		print("Using offset of {}".format(DS_Delay))
		print("Using Linear Interpolation.")
	elif DS_FLAG==2:
		print("Down sampling meal by rate of {}".format(DS_Rate))
		print("Using offset of {}".format(DS_Delay))
		print("Using Cubic Interpolation.")
	
		
	with open(DATABASE_FILEPATH,'rb') as fh:
		dataset = pkl.load(fh)
	OrigFreq = 64 # [Hz] Oreba data collection rate
	#OREBA_Cucumber={'README', 'UniqueID', 'proc_data', 'data_times', 'handedness',...
	#				'bites_gt', 'bites_handedness_gt'}
	
	
	proc = dataset['proc_data']
	biteGT_times=dataset['bites_gt']
	totalMeals = len(proc)
	
	# 0 = All Hands, 1 = Only Dominant, 2 = Only Non-Dominant
	if LIMIT_GT_HAND_FLAG != 0:
		# print warning to training in case data is being filtered for training to have permanent record
		if LIMIT_GT_HAND_FLAG == 1:
			print("Only Using DOMINANT Handed GT Intakes...")
		elif LIMIT_GT_HAND_FLAG == 2:
			print("Only Using LEFT Handed GT Intakes...")
		gtHandedness = dataset['bites_handedness_gt']
	# end of if LIMIT_HANDEDNESS_FLAG
	userHandedness = dataset['handedness']
	
	
	
	# Initialize Output
	training_data=[]
	classes=[]
	

	
	# Split the data into training and testing data
	meals_for_fold = TE_Training_Split(totalMeals, FOLD_SPLIT, FOLDS_TOTAL, FOLD_INDEX)


	for meal_num in meals_for_fold:

		meal_times=dataset['data_times'][meal_num]

		OG_meal_data=proc[meal_num][:,:]
		######### DOWN SAMPLING SECTION ###########
		if DS_FLAG==1: # Linear Interpolate Points to down sample and upsample
			DS_data=LinearInterp(OG_meal_data,DS_Rate,DS_Delay)
			meal_data=DS_data
		elif DS_FLAG==2: # Cubic interpolation
			DS_data=CubicInterp(OG_meal_data,DS_Rate,DS_Delay)
			meal_data=DS_data
		######### End of Downsampling Section #########
		else: # No Downsampling
			meal_data=OG_meal_data

		
		### RESAMPLE TO NEW FREQUENCY IF DESIRED ###
		if RESAMPLE_FLAG == 0: # if keeping original 15 Hz Signal
			meal_data = meal_data
			# Create times based on 0 index=0 and 15hz data
			meal_times = np.arange(len(meal_data))/float(OrigFreq)
		else: # if resampling data signal to a specific sampling frequency
			if True: # Cubic Interp
				meal_data = ResampleData(meal_data, OrigFreq, RESAMPLE_FLAG, 2)
			elif False: # Linear Interp
				meal_data = ResampleData(meal_data, OrigFreq, RESAMPLE_FLAG, 1)
			# Create times based on 0 index=0 and new frequency
			meal_times = np.arange(len(meal_data))/float(RESAMPLE_FLAG)
		# end of if resampling
		
		
		currMealBiteTimes=biteGT_times[meal_num]
		
		
		# LIMIT ONLY TO BITES OF HAND OF INTEREST
		if LIMIT_GT_HAND_FLAG != 0:
			currBiteHand=gtHandedness[meal_num] # for OREBA, 0 = Right, 1 = Left, 2 = Both
			currDomHand = 0 if userHandedness[meal_num] == 'right' else 1
			currNonDomHand = 0 if currDomHand==1 else 1 # Get non-dominant label as well
			if LIMIT_GT_HAND_FLAG == 1:
				#Strip the NonDominant Bites from GT LIST
				currMealBiteTimes=currMealBiteTimes[currBiteHand != currNonDomHand]
			elif LIMIT_GT_HAND_FLAG == 2:
				#Strip the NonDominant Bites from GT LIST
				currMealBiteTimes=currMealBiteTimes[currBiteHand != currNonDomHand]
			# end of reduced target
		# end of if LIMIT_GT_HAND_FLAG
		
		if True: # Using over 50% of a bite window as a "bite" and incrementing window count
			RequiredOverlapPercent = 0.5 # percentage of gt window that needs to be within cut
					# in order to be trigger incrementing window bite count
			for time_rover in range(0,len(meal_times)-CUT,STRIDE):

				WindowCnt = 0 # clear the window count each new training sample
				for i in range(len(currMealBiteTimes)): #check each Ground truth
					
					# if ((leftmost end of window) - (rightmost start of window)) > 0, then windows are overlapping 
					overlap=min(meal_times[time_rover+CUT], currMealBiteTimes[i,1]) - max(meal_times[time_rover], currMealBiteTimes[i,0]) 
					if overlap > 0:
						WindowTimeDuration=currMealBiteTimes[i,1]-currMealBiteTimes[i,0] # Grab current bite gt duration
						if overlap/WindowTimeDuration >= RequiredOverlapPercent: # if at least required overlap, then mark as bite and gt match
							WindowCnt+=1
				# end of for i in range(gt bites) loop
				
				# Append Window and its bite count
				training_data.append(meal_data[time_rover:time_rover+CUT,:])
				classes.append(WindowCnt)

			# end of time_rover loop

		if False: # Use percentage of bite gt window as the running sum of the bite count window  
			for time_rover in range(0,len(meal_times)-CUT,STRIDE):

				WindowCnt = 0.0 # clear the window count each new training sample
				
				for i in range(len(currMealBiteTimes)): #check each Ground truth
					# if ((leftmost end of window) - (rightmost start of window)) > 0, then windows are overlapping 
					overlap=min(meal_times[time_rover+CUT], currMealBiteTimes[i,1]) - max(meal_times[time_rover], currMealBiteTimes[i,0]) 
					if overlap > 0: # If there is overlap, add the portion of the gt_window in question to the count
						WindowTimeDuration=currMealBiteTimes[i,1]-currMealBiteTimes[i,0] # Grab current bite gt duration
						WindowCnt+=overlap/WindowTimeDuration
				# end of for i in range(gt bites) loop
				
				# Append Window and its bite count
				training_data.append(meal_data[time_rover:time_rover+CUT,:])
				classes.append(WindowCnt)

			# end of time_rover loop
		if False: #Use Robitic Bites of just learning the start of a window
			print("Unimplemented Bite Class Encoder...")

	#end of meal_num loop	


	np_classes=np.array(classes)
	np_training_data=np.array(training_data)

	return(np_training_data,np_classes)

# end of GenerateDataAndClasses_OREBA_WinCnt()
	
	
def GenerateDataAndClasses_ClemCafe_WinCnt(CUT, STRIDE, DATABASE_FILEPATH, FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=1, RESAMPLE_FLAG=0, SMOOTHING=0, LIMIT_GT_HAND_FLAG=0):
	'''
	Func: GenerateDataAndClasses_ClemCafe_WinCnt()
	Purpose: Connects Clemson Cafeteria Dataset to window count classes
		INPUTS
	CUT - Total number of Datapoints in a window of data
	STRIDE  - number of data points to move between each generated window
	DATABASE_FILES_NAMES - file address of the Pickle compressed data file, containing either ClemCafe Pickle
	FOLD_INDEX  - The fold number from [0,FOLDS_TOTAL)
	FOLDS_TOTAL - Either the # of folds (2+), or not dropping any data (0 or 1); leave one subject out not implemented on ClemCafe
	FOLD_SPLIT - either using block fold division (0) or striped division of files (1)
	RESAMPLE_FLAG - whether to keep original 15Hz data (flag=0) or to resample value of Resample Flag
	SMOOTHING - The number of points to smooth together
	LIMIT_GT_HAND_FLAG - Flag to limit the GT's targeted by the model;
					0 = All GT intakes (Both Hands), 1 = Dominant Hand Only, 2 = Non-Dominant Hand Only
		OUTPUTS
	np_training_data - windows of data that are CUT data wide and at least STRIDE data offset from each other
	np_classes - corresponding classes to each window in np_training_data, which is the number of bites in the window
	
	This Is all the information that was needed for converting Cafe data to SW Classifier. Build it up
	'''
	
	### Grab database ###
	with open(DATABASE_FILEPATH,'rb') as fh:
		dataset = pkl.load(fh)
	# ClemCafe_Cucumber={README,UniqueID,proc_data_rawAcc,proc_data_noG,handedness, ...
	#             ... bites_gt, bites_handedness_gt, means_global_rawAcc, ...
	#             ... stddev_global_rawAcc, means_global_noG, stddev_global_noG}

	OrigFreq=15.0 # original Cafeteria data is at 15Hz
	proc = dataset['proc_data_rawAcc']
	biteGT_indices=dataset['bites_gt']
	totalMeals = len(proc)
	handedness = dataset['handedness']
	
	# 0 = All Hands, 1 = Only Right, 2 = Only Left
	if LIMIT_GT_HAND_FLAG != 0:
		# print warning to training in case data is being filtered for training to have permanent record
		if LIMIT_GT_HAND_FLAG == 1:
			print("Only Using RIGHT Handed GT Intakes...")
		elif LIMIT_GT_HAND_FLAG == 2:
			print("Only Using LEFT Handed GT Intakes...")
		gt_handed_labels = dataset['bites_handedness_gt']
	# end of if LIMIT_HANDEDNESS_FLAG
	
	
	RequiredOverlapPercent = 0.8 # percentage of cut window that needs to be within bite_gt
					# in order to be marked as a bite class
	training_data=[]
	classes=[]

	# Split the data into training and testing data
	meals_for_fold = TE_Training_Split(totalMeals, FOLD_SPLIT, FOLDS_TOTAL, FOLD_INDEX)

	### Operate on each of the meals ###

	for meal_num in meals_for_fold:
		#Grab current meal gt bites and data
		biteGT_times = biteGT_indices[meal_num]/OrigFreq #convert to time domain to maintain data after resampling
		
		# LIMIT ONLY TO BITES OF HAND OF INTEREST
		if LIMIT_GT_HAND_FLAG != 0:
			currBiteHand = gt_handed_labels[meal_num] # for ClemCafe, 0 = Right, 1 = Left, 2 = Both
			currDomHand = handedness[meal_num]
			currNonDomHand = 0 if currDomHand==1 else 1 # Get non-dominant label as well
			if LIMIT_GT_HAND_FLAG == 1:
				#Strip the NonDominant Bites from GT LIST
				biteGT_times=biteGT_times[currBiteHand != currNonDomHand]
			elif LIMIT_GT_HAND_FLAG == 2:
				#Strip the NonDominant Bites from GT LIST
				biteGT_times=biteGT_times[currBiteHand != currNonDomHand]
			# end of reduced target
		# End of if LIMIT_GT_HAND_FLAG
		
		OG_meal_data=proc[meal_num][:,:]

		### RESAMPLE TO NEW FREQUENCY IF DESIRED ###

		if RESAMPLE_FLAG == 0: # if keeping original 15 Hz Signal
			meal_data = OG_meal_data
			# Create times based on 0 index=0 and 15hz data
			meal_data_times = np.arange(len(meal_data))/float(OrigFreq)
		else: # if resampling data signal to a specific sampling frequency
			if True: # Cubic Interp
				meal_data = ResampleData(OG_meal_data, OrigFreq, RESAMPLE_FLAG, 2)
			elif False: # Linear Interp
				meal_data = ResampleData(OG_meal_data, OrigFreq, RESAMPLE_FLAG, 1)
			# Create times based on 0 index=0 and new frequency
			meal_data_times = np.arange(len(meal_data))/float(RESAMPLE_FLAG)
		# end of if resampling



		### Parse Data and Count Bites ###

		for time_rover in range(0,len(meal_data_times)-CUT,STRIDE):

			WindowCnt = 0 # clear the window count each new training sample
			for i in range(len(biteGT_times)): #check each Ground truth
				if biteGT_times[i] > meal_data_times[time_rover] and biteGT_times[i] < meal_data_times[time_rover + CUT]:
					WindowCnt+=1
			# end of for i in range(gt bites) loop

			# Append Window and its bite count
			training_data.append(meal_data[time_rover:time_rover+CUT,:])
			classes.append(WindowCnt)

		# end of time_rover loop
	#end of meal_num loop	

	np_classes=np.array(classes)
	np_training_data=np.array(training_data)

	return(np_training_data,np_classes)
	
# end of GenerateDataAndClasses_ClemCafe_WinCn()






	
def GenerateDataAndClasses_ClemCafe_Bite(CUT, STRIDE, DATABASE_FILEPATH, FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=1, RESAMPLE_FLAG=0, SMOOTHING=0):
	'''
	Func: GenerateDataAndClasses_ClemCafe_Bite()
	Purpose: Connects Clemson Cafeteria Dataset to bite/non-bite classes
		INPUTS
	CUT
	STRIDE
	RAW_Data
	SMOOTHING
	RESAMPLE_FLAG - whether to keep original 15Hz data (flag=0) or to resample value of Resample Flag

		OUTPUTS
	np_training_data - windows of data that are CUT data wide and at least STRIDE data offset from each other
	np_classes - corresponding classes to each window in np_training_data, where 1=bite and 0=non-bite
	'''
	
	### Grab database ###
	with open(DATABASE_FILEPATH,'rb') as fh:
		dataset = pkl.load(fh)
	# ClemCafe_Cucumber={README,UniqueID,proc_data_rawAcc,proc_data_noG,handedness, ...
	#             ... bites_gt, bites_handedness_gt, means_global_rawAcc, ...
	#             ... stddev_global_rawAcc, means_global_noG, stddev_global_noG}

	OrigFreq=15.0 # original Cafeteria data is at 15Hz
	proc = dataset['proc_data_rawAcc']
	biteGT_indices=dataset['bites_gt']
	totalMeals = len(proc)

	training_data=[]
	classes=[]
	RequiredOverlapPercent = 0.8 # percentage of cut window that needs to be within bite_gt
					# in order to be marked as a bite class
	
	# Split the data into training and testing data
	meals_for_fold = TE_Training_Split(totalMeals, FOLD_SPLIT, FOLDS_TOTAL, FOLD_INDEX)

	### Operate on each of the meals ###

	for meal_num in meals_for_fold:
		#Grab current meal gt bites and data
		biteGT_times = biteGT_indices[meal_num]/OrigFreq #convert to time domain to maintain data after resampling
		OG_meal_data=proc[meal_num][:,:]

		### RESAMPLE TO NEW FREQUENCY IF DESIRED ###

		if RESAMPLE_FLAG == 0: # if keeping original 15 Hz Signal
			meal_data = OG_meal_data
			# Create times based on 0 index=0 and 15hz data
			meal_data_times = np.arange(len(meal_data))/float(OrigFreq)
		else: # if resampling data signal to a specific sampling frequency
			if True: # Cubic Interp
				meal_data = ResampleData(OG_meal_data, OrigFreq, RESAMPLE_FLAG, 2)
			elif False: # Linear Interp
				meal_data = ResampleData(OG_meal_data, OrigFreq, RESAMPLE_FLAG, 1)
			# Create times based on 0 index=0 and new frequency
			meal_data_times = np.arange(len(meal_data))/float(RESAMPLE_FLAG)
		# end of if resampling



		### Parse Data and Count Bites ###


		BiteWindows = np.zeros([len(biteGT_times),2])
		BiteWindows[:,0] = biteGT_times-3 # subtract 3 seconds from window for the start time
		BiteWindows[:,1] = biteGT_times+1 # add 1 second to window for the end time

		WindowTimeDuration=meal_data_times[0+CUT]-meal_data_times[0]
		for time_rover in range(0,len(meal_data_times)-CUT,STRIDE):

			InBiteFlag=-1 #clear the flag; -1 is no match, non-negative value is the location of match, -2 is error
			PartBiteFlag=0 # set to 1 if the window is more than 20% overlapping with a bite
			for i in range(len(BiteWindows)):
				# if ((leftmost end of window) - (rightmost start of window))>0, then windows are overlapping 
				overlap=min(meal_data_times[time_rover+CUT], BiteWindows[i,1]) - max(meal_data_times[time_rover], BiteWindows[i,0])
				if overlap > 0:
					# if at least required overlap, then mark as bite and gt match
					if overlap/WindowTimeDuration >= RequiredOverlapPercent:
						InBiteFlag=i
					# if greater than non-bite portion does not have required overlap percent
					elif overlap/WindowTimeDuration >= 1-RequiredOverlapPercent:
						PartBiteFlag=1
			# end of InBiteFlag check

			if InBiteFlag >= 0: # if a match was found, export positive class
				training_data.append(meal_data[time_rover:time_rover+CUT,:])
				classes.append(1)
			elif PartBiteFlag == 0: # If it didn't significantly overlap with a bite window
				training_data.append(meal_data[time_rover:time_rover+CUT,:])
				classes.append(0) # output negative class
			# else:
				# Do nothing since the window strattles clear data to avoid giving blurred examples to model

		# end of time_rover loop
	#end of meal_num loop	

	np_classes=np.array(classes)
	np_training_data=np.array(training_data)

	return(np_training_data,np_classes)


# end of GenerateDataAndClasses_ClemCafe_Bite()































def GenerateEvalData_OREBA(CUT, STRIDE, DATABASE_FILEPATH, FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=1, DS_FLAG=0, DS_Rate=1, DS_Delay=0, RESAMPLE_FLAG=0, SMOOTHING=0, GT_WINDOW_FLAG=0):
	'''
	Func: GenerateEvalData_OREBA()
	Purpose: Connects OREBA Dataset (both one and two hand) to evaluation windows
		INPUTS:
	CUT - Total number of Datapoints in a window of data
	STRIDE  - number of data points to move between each generated window
	DATABASE_FILES_NAMES - file address of the Pickle compressed data file, containing either one-handed or two-handed data
	FOLD_INDEX  - The fold number from [0,FOLDS_TOTAL)
	FOLDS_TOTAL - Either the number of folds (2+), or not dropping any data (0 or 1); leave one subject out not implemented on OREBA
	FOLD_SPLIT - either using block fold division (0) or striped division of files (1)
	DS_FLAG - whether to downsample data (1=LinearInterp, 2=CubicInterp) or to use the original signal (0)
	DS_Rate - # of points to combine when downsampling, then linearly interpolate.  
				e.g. DS_Rate=4 means that the signal will drop from 64 --> 16 Hz by grabbing every 4th data point
	DS_Delay - offset used in the selection of the data, in range [0,DS_Rate)
	RESAMPLE_FLAG - whether to keep original 15Hz data (flag=0) or to resample value of Resample Flag
		OUTPUTS:
	np_training_data - windows of data that are CUT data wide and at least STRIDE data offset from each other
	np_classes - corresponding classes to each window in np_training_data, which is the number of bites in the window
	'''
	# Set Stride and Cut to 1 if given zero because you need to at least move forward one datum
	if STRIDE == 0:
		STRIDE = 1
	if CUT == 0:
		CUT = 1
	
	# Output Message based on Down Sample Flag
	if DS_FLAG==1:
		print("Down sampling meal by rate of {}".format(DS_Rate))
		print("Using offset of {}".format(DS_Delay))
		print("Using Linear Interpolation.")
	elif DS_FLAG==2:
		print("Down sampling meal by rate of {}".format(DS_Rate))
		print("Using offset of {}".format(DS_Delay))
		print("Using Cubic Interpolation.")
	
	
	with open(DATABASE_FILEPATH,'rb') as fh:
		dataset = pkl.load(fh)
	#OREBA_Cucumber={'UniqueID','proc_data','handedness','bites_gt'}
	OrigFreq = 64 # [Hz] Oreba data collection rate
	
	proc = dataset['proc_data']
	biteGT_times=dataset['bites_gt']
	totalMeals = len(proc)
	
	training_data=[]
	classes=[]
	

	
	# Split the data into training and testing data
	meals_for_fold = TE_Eval_Split(totalMeals, FOLD_SPLIT, FOLDS_TOTAL, FOLD_INDEX)

	compiled_eval_data=[] # clear final output list

	for meal_num in meals_for_fold:
		
		meal_times=dataset['data_times'][meal_num]

		OG_meal_data=proc[meal_num][:,:]
		
		######### DOWN SAMPLING SECTION ###########
		if DS_FLAG==1: # Linear Interpolate Points to down sample and upsample
			DS_data=LinearInterp(OG_meal_data,DS_Rate,DS_Delay)
			meal_data=DS_data
		elif DS_FLAG==2: # Cubic interpolation
			DS_data=CubicInterp(OG_meal_data,DS_Rate,DS_Delay)
			meal_data=DS_data
		######### End of Downsampling Section #########
		else: # No Downsampling
			meal_data=OG_meal_data
		#end of if DS_FLAG switch statement
		
		### RESAMPLE TO NEW FREQUENCY IF DESIRED ###
		if RESAMPLE_FLAG == 0: # if keeping original 15 Hz Signal
			DataFreq = OrigFreq # The sampling rate of the data
			meal_data = meal_data
			# Create times based on 0 index=0 and 15hz data
			meal_times = np.arange(len(meal_data))/float(OrigFreq)
		else: # if resampling data signal to a specific sampling frequency
			DataFreq = RESAMPLE_FLAG # The sampling rate of the data
			if True: # Cubic Interp
				meal_data = ResampleData(meal_data, OrigFreq, RESAMPLE_FLAG, 2)
			elif False: # Linear Interp
				meal_data = ResampleData(meal_data, OrigFreq, RESAMPLE_FLAG, 1)
			# Create times based on 0 index=0 and new frequency
			meal_times = np.arange(len(meal_data))/float(RESAMPLE_FLAG)
		# end of if resampling
		
		### SMOOTH DATA IF DESIRED ###
		# averaging over a window centered on datum
		if SMOOTHING != 0:
			totalData,num_Sensors = np.shape(meal_data)
			SmoothedData=np.zeros(np.shape(meal_data))
			for i in range(totalData):
				for j in range(num_Sensors):
					SmoothedData[i,j]=meal_data[i,j]
			for i in range(SMOOTHING, totalData - SMOOTHING):
				for j in range(0, num_Sensors):
					total = 0.0
					for k in range(i-SMOOTHING, i+SMOOTHING+1):
						if (k >= 0 and k < totalData):
							total += meal_data[k,j]
					SmoothedData[i,j] = total / (SMOOTHING*2 + 1)
				# end of for j loop
			# end of for i loop
			meal_data=SmoothedData
		### END OF SMOOTHING ###
		
		
		
		
		##### PAD WITH ZEROS TO ENSURE FIRST WINDOW HAS NO BITES #####
		
		

		CUT_sec = CUT / DataFreq # obtain window cut and stride in the time domain
		STRIDE_sec = STRIDE / DataFreq

		# Give yourself three windows before having a bite enter the evaluator  
		MinBuffer = CUT_sec+2*STRIDE_sec
		
		
				
		LatestBite = max(biteGT_times[meal_num][:,1])
		EndEvalTime = LatestBite + MinBuffer
		if EndEvalTime > meal_times[-1]:
			NeededTime = EndEvalTime - meal_times[-1]
			NeededData = int(round(NeededTime*DataFreq+0.5)) # add 0.5 to force round up
			
			EndPaddingData = np.zeros([NeededData,len(meal_data[0])])
			EndPaddingTimes = np.arange(meal_times[-1]+1/float(DataFreq),
									 meal_times[-1]+NeededData*DataFreq,
									 1/float(DataFreq)
									)
			meal_data = np.append(meal_data,EndPaddingData,axis=0)
			meal_times = np.append(meal_times,EndPaddingTimes,axis=0)
			
			EndIndex = len(meal_data) #if addinbg was necessary, final index is the end of padding
		else:
			# end data one buffer after last bite event
			EndIndex = int((LatestBite+MinBuffer) * DataFreq)
		# end if Need to Pad End with Zeros
		
		
		EarliestBite=min(biteGT_times[meal_num][:,0]) # grab the earliest start of a bite
		StartEvalTime = EarliestBite - MinBuffer
		# If negative start time, add needed number of entries
		if StartEvalTime < 0:
			NeededTime = -StartEvalTime
			NeededData = int(round(NeededTime*DataFreq+0.5)) # add 0.5 to force round up
			NeededData -= 1 #subtract one for the zero time entry already in data

			StartPaddingData = np.zeros([NeededData,len(meal_data[0])])
			StartPaddingTimes = np.arange(-(NeededData/DataFreq),0,1/float(DataFreq))

			meal_data=np.append(StartPaddingData, meal_data, axis=0)
			meal_times=np.append(StartPaddingTimes, meal_times, axis=0)
			
			# Set start time to very beginning of data
			StartIndex = 0
			# ADJUST END INDEX TO ACCOUNT FOR BEGINNING PADDING 
			EndIndex += len(StartPaddingTimes)
		else:
			# Set Start index to start MinBuffer Before first bite
			StartIndex = int(StartEvalTime*DataFreq)
		# end if Need to Pad Start with Zeros

		
		# determine StartTime
		eval_data=[] # output 6-axis data normalize
		timestep_ref=[] # Collection of the first timesteps for each window
		for time_rover in range(StartIndex,EndIndex-CUT,STRIDE):
			eval_data.append(meal_data[time_rover:time_rover+CUT,:])
			timestep_ref.append(meal_times[time_rover])

		np_eval_data=np.array(eval_data)
		
		if GT_WINDOW_FLAG == 0: # if grabbing a single index of time, grab midpoint
			currBiteGT_times = (biteGT_times[meal_num][:,0]+biteGT_times[meal_num][:,1])/2
		else: # if grabbing the full window
			currBiteGT_times = biteGT_times[meal_num]
		# end of if Window or not switch
		
		compiled_eval_data.append([meal_num,np_eval_data,timestep_ref,currBiteGT_times])
	# end of for meal_num loop

	return compiled_eval_data

# end of GenerateEvalData_OREBA()



def GenerateEvalData_ClemCafe(CUT, STRIDE, DATABASE_FILEPATH, FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=1, RESAMPLE_FLAG=0, SMOOTHING=0, GT_WINDOW_FLAG=0):
	'''
	Func: GenerateDataAndClasses_ClemCafe_WinCnt()
	Purpose: Connects Clemson Cafeteria Dataset to evaluation windows
		INPUTS
	CUT - Total number of Datapoints in a window of data
	STRIDE  - number of data points to move between each generated window
	DATABASE_FILES_NAMES - file address of the Pickle compressed data file, containing either ClemCafe Pickle
	FOLD_INDEX  - The fold number from [0,FOLDS_TOTAL)
	FOLDS_TOTAL - Either the # of folds (2+), or not dropping any data (0 or 1); leave one subject out not implemented on ClemCafe
	FOLD_SPLIT - either using block fold division (0) or striped division of files (1)
	RESAMPLE_FLAG - whether to keep original 15Hz data (flag=0) or to resample value of Resample Flag
	SMOOTHING - The number of points to smooth together
		OUTPUTS
	np_training_data - windows of data that are CUT data wide and at least STRIDE data offset from each other
	np_classes - corresponding classes to each window in np_training_data, which is the number of bites in the window
	
	This Is all the information that was needed for converting Cafe data to SW Classifier. Build it up
	'''
	
	# Set Stride and Cut to 1 if given zero because you need to at least move forward one datum
	if STRIDE == 0:
		STRIDE = 1
	if CUT == 0:
		CUT = 1
	
	### Grab database ###
	with open(DATABASE_FILEPATH,'rb') as fh:
		dataset = pkl.load(fh)
	# ClemCafe_Cucumber={README,UniqueID,proc_data_rawAcc,proc_data_noG,handedness, ...
	#             ... bites_gt, bites_handedness_gt, means_global_rawAcc, ...
	#             ... stddev_global_rawAcc, means_global_noG, stddev_global_noG}

	
	LIMIT_HANDEDNESS_FLAG = 0 # 0 = All Hands, 1 = Only Right, 2 = Only Left
	if LIMIT_HANDEDNESS_FLAG != 0:
		# print warning to training in case data is being filtered for training to have permanent record
		if LIMIT_HANDEDNESS_FLAG == 1:
			print("Only Using RIGHT Handed Monitered Meals...")
		elif LIMIT_HANDEDNESS_FLAG == 2:
			print("Only Using LEFT Handed Monitered Meals...")
		HandednessList = dataset['handedness']
	# end of if LIMIT_HANDEDNESS_FLAG
	
	OrigFreq=15.0 # original Cafeteria data is at 15Hz
	if True:
		proc = dataset['proc_data_rawAcc']
	else:
		proc = dataset['proc_data_noG']
	biteGT_indices=dataset['bites_gt']
	totalMeals = len(proc)
	
	training_data=[]
	classes=[]
	

	
	# Split the data into training and testing data
	meals_for_fold = TE_Eval_Split(totalMeals, FOLD_SPLIT, FOLDS_TOTAL, FOLD_INDEX)

	compiled_eval_data=[] # clear final output list

	for meal_num in meals_for_fold:
		
		if LIMIT_HANDEDNESS_FLAG == 1:
			if HandednessList[meal_num] == 1: # if meal is marked as a lefty, skip
				continue
		elif LIMIT_HANDEDNESS_FLAG == 2:
			if HandednessList[meal_num] == 0: # if meal is marked as a lefty, skip
				continue
		# end of if LIMIT_HANDEDNESS_FLAG
		
		
		currBiteGT_times=biteGT_indices[meal_num]/OrigFreq
		if GT_WINDOW_FLAG == 0: # if grabbing a single index of time, make window have start and end as the given index
			# Necessary to keep references later concise regardless of GT_WINDOW_FLAG
			currBiteGT_times_Windows = np.zeros([len(currBiteGT_times),2])
			currBiteGT_times_Windows[:,0] = currBiteGT_times
			currBiteGT_times_Windows[:,1] = currBiteGT_times
		else: # if grabbing the full window
			currBiteGT_times_Windows = np.zeros([len(currBiteGT_times),2])
			currBiteGT_times_Windows[:,0] = currBiteGT_times-2.5 # subtract 2.5 seconds from intake moment for motion leading to bite
			currBiteGT_times_Windows[:,1] = currBiteGT_times+1.5 # add 1.5 seconds after intake moment for motion after bite
		# end of if Window or not switch
		
		
		
		OG_meal_data=proc[meal_num][:,:]

		if RESAMPLE_FLAG == 0: # if keeping original 15 Hz Signal
			meal_data = OG_meal_data
			# Create times based on 0 index=0 and 15hz data
			DataFreq = OrigFreq # Keep original sampling rate of the data
			meal_times = np.arange(len(meal_data))/float(DataFreq)
		else: # if resampling data signal to a specific sampling frequency
			if True: # Cubic Interp
				meal_data = ResampleData(OG_meal_data, OrigFreq, RESAMPLE_FLAG, 2)
			elif False: # Linear Interp
				meal_data = ResampleData(OG_meal_data, OrigFreq, RESAMPLE_FLAG, 1)
			# Create times based on 0 index=0 and new frequency
			DataFreq = RESAMPLE_FLAG
			meal_times = np.arange(len(meal_data))/float(DataFreq)
		# end of if resampling
		
		### SMOOTH DATA IF DESIRED ###
		# averaging over a window centered on datum
		if SMOOTHING != 0:
			totalData,num_Sensors = np.shape(meal_data)
			SmoothedData=np.zeros(np.shape(meal_data))
			for i in range(totalData):
				for j in range(num_Sensors):
					SmoothedData[i,j]=meal_data[i,j]
			for i in range(SMOOTHING, totalData - SMOOTHING):
				for j in range(0, num_Sensors):
					total = 0.0
					for k in range(i-SMOOTHING, i+SMOOTHING+1):
						if (k >= 0 and k < totalData):
							total += meal_data[k,j]
					SmoothedData[i,j] = total / (SMOOTHING*2 + 1)
				# end of for j loop
			# end of for i loop
			meal_data=SmoothedData
		### END OF SMOOTHING ###
		
		


		##### PAD WITH ZEROS TO ENSURE FIRST WINDOW HAS NO BITES #####


		CUT_sec = CUT / DataFreq # obtain window cut and stride in the time domain
		STRIDE_sec = STRIDE / DataFreq

		# Give yourself three windows before having a bite enter the evaluator  
		MinBuffer = CUT_sec+2*STRIDE_sec
		
		
				
		LatestBite = max(currBiteGT_times_Windows[:,1])
		EndEvalTime = LatestBite + MinBuffer
		if EndEvalTime > meal_times[-1]:
			NeededTime = EndEvalTime - meal_times[-1]
			NeededData = int(round(NeededTime*DataFreq+0.5)) # add 0.5 to force round up
			
			EndPaddingData = np.zeros([NeededData,len(meal_data[0])])
			EndPaddingTimes = np.arange(meal_times[-1]+1/float(DataFreq),
									 meal_times[-1]+NeededData*DataFreq,
									 1/float(DataFreq)
									)
			meal_data = np.append(meal_data,EndPaddingData,axis=0)
			meal_times = np.append(meal_times,EndPaddingTimes,axis=0)
			
			EndIndex = len(meal_data) #if addinbg was necessary, final index is the end of padding
		else:
			# end data one buffer after last bite event
			EndIndex = int((LatestBite+MinBuffer) * DataFreq)
		# end if Need to Pad End with Zeros
		
		
		EarliestBite=min(currBiteGT_times_Windows[:,0]) # grab the earliest start of a bite
		StartEvalTime = EarliestBite - MinBuffer
		# If negative start time, add needed number of entries
		if StartEvalTime < 0:
			NeededTime = -StartEvalTime
			NeededData = int(round(NeededTime*DataFreq+0.5)) # add 0.5 to force round up
			NeededData -= 1 #subtract one for the zero time entry already in data

			StartPaddingData = np.zeros([NeededData,len(meal_data[0])])
			StartPaddingTimes = np.arange(-(NeededData/DataFreq),0,1/float(DataFreq))

			meal_data=np.append(StartPaddingData, meal_data, axis=0)
			meal_times=np.append(StartPaddingTimes, meal_times, axis=0)
			
			# Set start time to very beginning of data
			StartIndex = 0
			# ADJUST END INDEX TO ACCOUNT FOR BEGINNING PADDING 
			EndIndex += len(StartPaddingTimes)
		else:
			# Set Start index to start MinBuffer Before first bite
			StartIndex = int(StartEvalTime*DataFreq)
		# end if Need to Pad Start with Zeros
		
		
		
		eval_data=[] # output 6-axis data normalize
		timestep_ref=[] # Collection of the first timesteps for each window
		for time_rover in range(StartIndex,EndIndex-CUT,STRIDE):
			eval_data.append(meal_data[time_rover:time_rover+CUT,:])
			timestep_ref.append(meal_times[time_rover])

		np_eval_data=np.array(eval_data)
		
		
		if GT_WINDOW_FLAG == 0: # if grabbing a single index of time, grab midpoint
			BiteGT_Output = currBiteGT_times
		else: # if grabbing the full window
			BiteGT_Output = currBiteGT_times_Windows
		# end of if Window or not switch
		
		compiled_eval_data.append([meal_num,np_eval_data,timestep_ref,BiteGT_Output])
	# end of for meal_num loop

	return compiled_eval_data
	
# end of GenerateEvalData_ClemCafe()



def GenerateTallyCounts(GT_List, Timestamps, CUT_sec): 
	'''
	Func: GenerateTallyCounts()
	Purpose: Produces a list GT event counts corresponding to each time in Timetamps with a reach of size CUT_sec of total events (from GT_List)
		INPUTS:
	GT_List - 1D list or numpy array of the ground truth moments
	Timestamps - 1D array the starting time of each window for which to calculate the tallys
	CUT_sec - The size of a window within which to tally GT events
		OUTPUTS:
	EventCounts - 1D list of the total number of events in each window 
	'''
	
	# ensure GT_List is an np.array
	GT_List = np.array(GT_List)
	
	
	### Create Tallys for each timestamp
	EventCounts=[] # initialize event counts
	for currTime in Timestamps:
		# Sum the TRUE values which are within the window 
		EventCounts.append(np.sum(np.logical_and(GT_List > currTime, GT_List < currTime + CUT_sec))) 
	# end of for currTime
	
	
	return(EventCounts)
	
	
# end of def GenerateTallyCounts()











