'''
File: ClassDataFuncs.py
Written by: James Jolly
Purpose: House all functions used for evaluating event detections vs ground truths.
		While currently using three datasets (Clemson Cafeteria, Oreba 1Hand, and Oreba 2Hand), these 
		functions are meant to handle time point predictions and ground truths
		and different model Ground truths (bite/no bite versus a total count in the window).
TABLE of CONTENTS
	CountsToDetections
		Cnt2Detect_MidpointTriggerMidpointPlacement()
		Cnt2Detect_TetrisBlockFilling()

		

Segments
'''


import numpy as np
import scipy
from scipy import interpolate
import math

from io import UnsupportedOperation
import pickle as pkl





#############################################################
###########   Prediction To Detection FUNCTIONS   ###########
#############################################################


def Cnt2Detect_MidpointTriggerMidpointPlacement(windowCounts, timesteps, STRIDE_sec, WINDOW_sec):
	'''
	Func: Pred2Detect_MidpointTriggerMidpointPlacement()
	Purpose: Takes a single meal worth of (TallyNet Count predicitions),
			the timestamps of each of these windows, and the window size and stride,
			then calculates a list of detections times for all events in the counts.
			  Method scales the running summation across the counts, then triggers an
			event whenever the sum rounds to the next integer.  Triggered detections are 
			either placed with a constant offset (PLACEMENT_METHOD_FLAG=1, discouraged), 
			or at the midpoint of the current window (PLACEMENT_METHOD_FLAG=2, recommended).
	INPUTS
		windowCounts - 1D numpy array [counts]
			List of predicted window counts across a meal
		timesteps - 1D numpy array [sec]
			List of Timestamps, corresponding to the start of each window in windowCounts, in seconds 
		stride_sec - Float
			Size of steps between window entries in windowCounts
		window_sec - Float
			Duration of each window in seconds 
	OUTPUT
		Detections - 1D Numpy array [sec]
			List of all detections
	'''
	# # # INITIALIZE CONSTANTS # # #
	DEBUG_PRINTS = 0
	PLACEMENT_METHOD_FLAG=2 # Variable used to signal how and where event is placed in window
		# 1 = hardcoded 300 second offset (used until 9/14/21)
		# 2 = adaptive midpoint of window (simple method used by Basil in MS thesis)
	PLACEMENT_METHOD_1_OFFSET = 10 # time in [sec] for hard offset placement method
	Diff=0.001
	
	# Initialize Variables used 
	FloatBites = 0.0
	TotalBites = 0
	
	
	Detections=[] # initialize as empty list, add detections as they are found
	for i in range(len(windowCounts)):
		bites_in_window = float(windowCounts[i])
		if (bites_in_window < 0.0):
			bites_in_window = 0.0

		FloatBites = FloatBites + (bites_in_window / WINDOW_sec * STRIDE_sec)
		if (TotalBites < round(FloatBites)):
			TotalBites += 1

			if PLACEMENT_METHOD_FLAG == 1:
				Detections.append(timesteps[i]+PLACEMENT_METHOD_1_OFFSET)
			elif PLACEMENT_METHOD_FLAG == 2:

				Spot = timesteps[i]+(WINDOW_sec/2)
				if Spot <= 0:
					if DEBUG_PRINTS == 1:
						print("negative Prediction: Meal Num = {}, Fold {}".format(meal_num,FOLD_INDEX))
					#end of DEBUG_PRINTS
					Spot = Diff
					Diff += 0.001
				# end of negative check
				Detections.append(Spot)
			else:
				print('Please select a placement method for locating events within window.')
				exit(0)
			# end of Placement Flag switch
	# end of i in range(windowCounts) loop
	
	return Detections

# end of def Cnt2Detect_MidpointTriggerMidpointPlacement


def Cnt2Detect_TetrisBlockFilling(windowCounts, timesteps, STRIDE_sec, WINDOW_sec):
	'''
	Func: Pred2Detect_TetrisBlockFilling()
	Purpose: Takes a single meal worth of (TallyNet Count predicitions),
			the timestamps of each of these windows, and the window size and stride,
			then calculates a list of detections times for all events in the counts.
			  Method keeps a list of buckets tracking windowed events.  Each count is
			added to the buckets, with a max of 1.0 being added to the first bucket,
			then another 1.0 to the next bucket, continually "spilling over" into 
			however many buckets are needed until the count is reached. 
			  When the first bucket has reached a count of WINDOW_sec/STRIDE_sec,
			which is the number of windows an event would appear inside, the detection 
			is triggered and placed at the current windows timestamp.
	INPUTS
		windowCounts - 1D numpy array [counts]
			List of predicted window counts across a meal
		timesteps - 1D numpy array [sec]
			List of Timestamps, corresponding to the start of each window in windowCounts, in seconds 
		stride_sec - Float
			Size of steps between window entries in windowCounts
		window_sec - Float
			Duration of each window in seconds 
	OUTPUT
		Detections - 1D Numpy array [sec]
			List of all detections
	********* NOTE *********
		Potential bug losing some of the count if TriggerTarget is not an 
	'''
	
	
	### INPUT VALIDATION ###
	
	
	### DEFINE VARIABLES ###
	MAX_COUNT = int(np.ceil(max(windowCounts)))
	hist_buckets = np.zeros(MAX_COUNT+1) # blank list of buckets, with a spare
	TriggerTarget = WINDOW_sec/STRIDE_sec

	TotalBites = 0

	Diff=0.001 # minimum spacing for bites triggered at the same time


	# # # # # Use model Predictions to place bite in Window # # # # #


	Detections=[] # initialize as empty list, add detections as they are found
	NegativeTimeDetectCnt = 0 # used in rare case of negative indices of time
	for i in range(len(windowCounts)):
		bites_in_window = float(windowCounts[i])
		if (bites_in_window < 0.0):
			bites_in_window = 0.0
		# end of if negative bites

		currBucket = 0 # cntr
		while bites_in_window >= 1.0:
			hist_buckets[currBucket] += 1.0 # add to bucket
			bites_in_window -= 1.0 # take away from cnt
			currBucket += 1 # increment to "spill over" into next bucket 
		# end while 
		hist_buckets[currBucket] += bites_in_window # add remaining amount

		# Check for detections
		DiffCnt = 0 # counter used only if multiple detections trigger at the same time (unlikely)
		
		while hist_buckets[0] >= TriggerTarget:
			Spot = timesteps[i] + Diff * DiffCnt
			if Spot < Diff * NegativeTimeDetectCnt:
				NegativeTimeDetectCnt += 1
				Spot = Diff * NegativeTimeDetectCnt
			Detections.append(Spot)
			DiffCnt += 1 # increment DiffCnt in case futher detections happen

			# Spill over remaining, then shift buckets
			hist_buckets[1] += (hist_buckets[0] - TriggerTarget)
			for i in range(len(hist_buckets) - 1):
				hist_buckets[i] = hist_buckets[i+1]
			# end for i
			hist_buckets[-1] = 0 # clear last bucket
		# end while 

	# end of i in range(windowCounts) loop
	
	return Detections
	
# end of def Cnt2Detect_TetrisBlockFilling()


def Cnt2Detect_TemplateMatching(windowCounts, timesteps, STRIDE_sec, WINDOW_sec, MATCH_THRESH = 0.95):
	'''
	Func: Pred2Detect_TetrisBlockFilling()
	Purpose: Takes a single meal worth of (TallyNet Count predicitions),
			the timestamps of each of these windows, and the window size and stride,
			then calculates a list of detections times for all events in the counts.
			  Method keeps a list of buckets tracking windowed events.  Each count is
			added to the buckets, with a max of 1.0 being added to the first bucket,
			then another 1.0 to the next bucket, continually "spilling over" into 
			however many buckets are needed until the count is reached. 
			  When the first bucket has reached a count of WINDOW_sec/STRIDE_sec,
			which is the number of windows an event would appear inside, the detection 
			is triggered and placed at the current windows timestamp.
	INPUTS
		windowCounts - 1D numpy array [counts]
			List of predicted window counts across a meal
		timesteps - 1D numpy array [sec]
			List of Timestamps, corresponding to the start of each window in windowCounts, in seconds 
		stride_sec - Float
			Size of steps between window entries in windowCounts
		window_sec - Float
			Duration of each window in seconds,
		MATCHING_THRESH - float (0.0 to 1.0]
			The score of the template match to trigger a detection
	OUTPUT
		Detections - 1D Numpy array [sec]
			List of all detections
	********* NOTE *********
		Potential bug losing some of the count if TriggerTarget is not an 
	'''
	
	
	### INPUT VALIDATION ###
	
	# Copy inputs to local versions so they can all be modified without impacting outside scope of function
	windowCounts_local = windowCounts.copy()

	### DEFINE VARIABLES ###
	# Adjustments to make based on previous detections
	ExpectedCnts = np.zeros(len(windowCounts))
	# Create Capped counts to eliminate impact on template match based on overlap
	CappedCnts = windowCounts.copy()
	CappedCnts[CappedCnts > 1.0] = 1.0 # cap to single detection
	# MAY NOT BE NEEDED # Not doing Causes negative dips to reduce score 
	CappedCnts[CappedCnts < 0.0] = 0.0 # eliminate negative


	# Template is a square wave with height of 1.0 of length Template Size
	TemplateSize = int(np.ceil(WINDOW_sec/STRIDE_sec))
	Template = np.ones(TemplateSize)

	Diff=0.001 # minimum spacing for bites triggered at the same time
	NegativeTimeDetectCnt = 0 # used in rare case of negative indices of time


	# # # # # Use model Predictions to place bite in Window # # # # #

	Detections=[] # initialize as empty list, add detections as they are found

	for i in range(len(windowCounts_local)-TemplateSize):

		# Compute Template Match Score for current index
		TemplateScore = 0.0
		for j in range(TemplateSize):
			TemplateScore += CappedCnts[i+j] * Template[j] # cnt multiplied by template
		# end of for j
		TemplateScore /= TemplateSize # scale by size to create score [0.0, 1.0]
		# print(TemplateScore)

		if TemplateScore >= MATCH_THRESH:
			Detections.append(timesteps[i+TemplateSize-1])

			# adjust capped cnts
			for j in range(TemplateSize):
				windowCounts_local[i+j] -= Template[j] # take away from original counts
				newCappedCnt = windowCounts_local[i+j] # update the capped Count
				if newCappedCnt > 1.0: # cap value if needed
					newCappedCnt = 1.0
				# end if newCappedCnt

				CappedCnts[i+j] = newCappedCnt
				### MAY ADD # Potentially cap to make non-negative
			# end of for j
		# end of if TemplateScore
	# end of i in range(windowCounts) loop
	
	return Detections
	
# end of def Cnt2Detect_TetrisBlockFilling()








def Cnt2Detect_TetrisBlockFilling_Midpt(windowCounts, timesteps, STRIDE_sec, WINDOW_sec):
	'''
	Func: Pred2Detect_TetrisBlockFilling()
	Purpose: Takes a single meal worth of (TallyNet Count predicitions),
			the timestamps of each of these windows, and the window size and stride,
			then calculates a list of detections times for all events in the counts.
			  Method keeps a list of buckets tracking windowed events.  Each count is
			added to the buckets, with a max of 1.0 being added to the first bucket,
			then another 1.0 to the next bucket, continually "spilling over" into 
			however many buckets are needed until the count is reached. 
			  When the first bucket has reached a count of WINDOW_sec/STRIDE_sec,
			which is the number of windows an event would appear inside, the detection 
			is triggered and placed at the current windows timestamp.
	INPUTS
		windowCounts - 1D numpy array [counts]
			List of predicted window counts across a meal
		timesteps - 1D numpy array [sec]
			List of Timestamps, corresponding to the start of each window in windowCounts, in seconds 
		stride_sec - Float
			Size of steps between window entries in windowCounts
		window_sec - Float
			Duration of each window in seconds 
	OUTPUT
		Detections - 1D Numpy array [sec]
			List of all detections
	********* NOTE *********
		Potential bug losing some of the count if TriggerTarget is not an 
	'''
	
	
	### INPUT VALIDATION ###
	

	### DEFINE VARIABLES ###
	MAX_COUNT = int(np.ceil(max(windowCounts)))
	hist_buckets = np.zeros(MAX_COUNT+1) # blank list of buckets, with a spare
	TriggerTarget = WINDOW_sec/(2*STRIDE_sec) # Trigger at the half way point
	ClearTarget = WINDOW_sec/STRIDE_sec
	TriggeredNotClearedFlag = (hist_buckets != 0)
	PlacementOffset = WINDOW_sec/2

	TotalBites = 0

	Diff=0.001 # minimum spacing for bites triggered at the same time


	# # # # # Use model Predictions to place bite in Window # # # # #


	Detections=[] # initialize as empty list, add detections as they are found
	NegativeTimeDetectCnt = 0 # used in rare case of negative indices of time
	for i in range(len(windowCounts)):
		bites_in_window = float(windowCounts[i])
		if (bites_in_window < 0.0):
			bites_in_window = 0.0
		# end of if negative bites

		currBucket = 0 # cntr
		while bites_in_window >= 1.0:
			hist_buckets[currBucket] += 1.0 # add to bucket
			bites_in_window -= 1.0 # take away from cnt
			currBucket += 1 # increment to "spill over" into next bucket 
		# end while 
		hist_buckets[currBucket] += bites_in_window # add remaining amount

		# Check for detections
		DiffCnt = 0 # counter used only if multiple detections trigger at the same time (unlikely)

		# for j in range(len(hist_buckets)):
		j = 0
		while j < len(hist_buckets):
			if ((TriggeredNotClearedFlag[j] == False) and (hist_buckets[j] >= TriggerTarget)):
				Spot = timesteps[i] + PlacementOffset + Diff * DiffCnt
				if Spot < Diff * NegativeTimeDetectCnt:
					NegativeTimeDetectCnt += 1
					Spot = Diff * NegativeTimeDetectCnt
				Detections.append(Spot)
				DiffCnt += 1 # increment DiffCnt in case futher detections happen

				TriggeredNotClearedFlag[j] = True
			#end if Trigger

			if hist_buckets[j] >= ClearTarget:
				# Spill over remaining, then shift buckets
				hist_buckets[1] += (hist_buckets[0] - ClearTarget)
				for k in range(len(hist_buckets) - 1):
					hist_buckets[k] = hist_buckets[k+1]
					TriggeredNotClearedFlag[k] = TriggeredNotClearedFlag[k+1]
				# end for i
				hist_buckets[-1] = 0 # clear last bucket
				TriggeredNotClearedFlag[-1] = False
				j -= 1 # subtract from j to repeat index now that all hist_buckets shifted over
			# end while

			j += 1
		# end of while j < len(hist_buckets)

	# end of i in range(windowCounts) loop

	return (Detections)

# end of Cnt2Detect_TetrisBlockFilling_Midpt()






### YET TO BE IMPLEMENTED ###
def Cnt2Detect_TetrisBlockFilling_Decay(windowCounts, timesteps, STRIDE_sec, WINDOW_sec):
	'''
	Func: Pred2Detect_TetrisBlockFilling()
	Purpose: Takes a single meal worth of (TallyNet Count predicitions),
			the timestamps of each of these windows, and the window size and stride,
			then calculates a list of detections times for all events in the counts.
			  Method keeps a list of buckets tracking windowed events.  Each count is
			added to the buckets, with a max of 1.0 being added to the first bucket,
			then another 1.0 to the next bucket, continually "spilling over" into 
			however many buckets are needed until the count is reached. 
			  When the first bucket has reached a count of WINDOW_sec/STRIDE_sec,
			which is the number of windows an event would appear inside, the detection 
			is triggered and placed at the current windows timestamp.
	INPUTS
		windowCounts - 1D numpy array [counts]
			List of predicted window counts across a meal
		timesteps - 1D numpy array [sec]
			List of Timestamps, corresponding to the start of each window in windowCounts, in seconds 
		stride_sec - Float
			Size of steps between window entries in windowCounts
		window_sec - Float
			Duration of each window in seconds 
	OUTPUT
		Detections - 1D Numpy array [sec]
			List of all detections
	********* NOTE *********
		Potential bug losing some of the count if TriggerTarget is not an 
	'''
	### INPUT VALIDATION ###
	
    
    
    
	### INPUT VALIDATION ###
	windowCounts = Predictions.copy()
	timesteps = Timesteps.copy()

	### DEFINE VARIABLES ###
	MAX_COUNT = int(np.ceil(max(windowCounts)))
	hist_buckets = np.zeros(MAX_COUNT+1) # blank list of buckets, with a spare
	TriggerTarget = WINDOW_sec/(2*STRIDE_sec) # Trigger at the half way point
	ClearTarget = WINDOW_sec/STRIDE_sec
	TriggeredNotClearedFlag = (hist_buckets != 0)
	PlacementOffset = WINDOW_sec/2

	TotalBites = 0

	Diff=0.001 # minimum spacing for bites triggered at the same time


	# # # # # Use model Predictions to place bite in Window # # # # #


	Detections=[] # initialize as empty list, add detections as they are found
	NegativeTimeDetectCnt = 0 # used in rare case of negative indices of time
	for i in range(len(windowCounts)):
		bites_in_window = float(windowCounts[i])
		if (bites_in_window < 0.0):
			bites_in_window = 0.0
		# end of if negative bites

		currBucket = 0 # cntr
		while bites_in_window >= 1.0:
			hist_buckets[currBucket] += 1.0 # add to bucket
			bites_in_window -= 1.0 # take away from cnt
			currBucket += 1 # increment to "spill over" into next bucket 
		# end while 
		hist_buckets[currBucket] += bites_in_window # add remaining amount

		# Check for detections
		DiffCnt = 0 # counter used only if multiple detections trigger at the same time (unlikely)

		# for j in range(len(hist_buckets)):
		j = 0
		while j < len(hist_buckets):
			if ((TriggeredNotClearedFlag[j] == False) and (hist_buckets[j] >= TriggerTarget)):
				Spot = timesteps[i] + PlacementOffset + Diff * DiffCnt
				if Spot < Diff * NegativeTimeDetectCnt:
					NegativeTimeDetectCnt += 1
					Spot = Diff * NegativeTimeDetectCnt
				Detections.append(Spot)
				DiffCnt += 1 # increment DiffCnt in case futher detections happen

				TriggeredNotClearedFlag[j] = True
			#end if Trigger

			if hist_buckets[j] >= ClearTarget:
				# Spill over remaining, then shift buckets
				hist_buckets[1] += (hist_buckets[0] - ClearTarget)
				for k in range(len(hist_buckets) - 1):
					hist_buckets[k] = hist_buckets[k+1]
					TriggeredNotClearedFlag[k] = TriggeredNotClearedFlag[k+1]
				# end for i
				hist_buckets[-1] = 0 # clear last bucket
				TriggeredNotClearedFlag[-1] = False
				j -= 1 # subtract from j to repeat index now that all hist_buckets shifted over
			# end while

			j += 1
		# end of while j < len(hist_buckets)

	# end of i in range(windowCounts) loop

	return (Detections)

# end of Cnt2Detect_TetrisBlockFilling_MidptDecay()







#####################################################
###########       TEMPLATE FUNCTION       ###########
#####################################################


def FuncName(Inputs):
	'''
	Func: FuncName()
	Purpose: Description
		INPUTS
	Inputs   - Description
		OUTPUT
	Outputs  - Description
	'''
	
	
	# Return Final Results
	return Output
# end of FuncName()

#####################################################
###########        END OF TEMPLATE        ###########
#####################################################





