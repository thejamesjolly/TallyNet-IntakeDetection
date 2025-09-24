'''
File: ClassDataFuncs.py
Written by: James Jolly
Purpose: House all functions used for evaluating event detections vs ground truths.
		While currently using three datasets (Clemson Cafeteria, Oreba 1Hand, and Oreba 2Hand), these 
		functions are meant to handle time point predictions and ground truths
		and different model Ground truths (bite/no bite versus a total count in the window).
TABLE of CONTENTS
	Point Window Conversion Funcs
		TimePoint2Window()
		Window2TimePoint()
	CountsToDetections
		Cnt2Detect_MidpointTriggerMidpointPlacement()
		Cnt2Detect_TetrisBlockFilling()
	Evaluation Functions
		DongEval_PtVsPt()
		KyritEval_PtVsWindow()
		Duration_WindowVsWindow()
		Section_WindowVsWindow()
	Printing Results
		PrintStats_SingleLine()
		PrintDetailedLine()

		

Segments
'''


import numpy as np
import scipy
from scipy import interpolate
import math

from io import UnsupportedOperation
import pickle as pkl



#############################################################
###########   Point Window Conversion FUNCTIONS   ###########
#############################################################

def TimePoint2Window(TimePoints, WindowSz = 2.5, Offset = 0.5, DELTA_DURATION = 0.001):
	'''
	Func: TimePoint2Window()
	Purpose: Takes specific points of time (predictions, GT, etc.) and creates a window of time for each of these predictions.
			It allows for tuning the size of the windows made and the placement of window about the event.
		### WARNING ###:
	This function does not check if windows are overlapping once made.
		INPUTS
	TimePoints - (np_array)   [sec]   shape = (Nx1)
			Input sequence of timesteps to convert.  Assumed to be in seconds, and non-negative.
	WindowSz   - (float)   [sec]   Default = 2.5, Range (0.0, inf)
			Length of time for the final window. All windows will be the same size.
			Defaults to the average size of a bite (~2.5 sec)
	Offset     - (float)   [%]   Default = 0.5, Range = [0.0, 1.0]
			Where to place the event relative to the start (0.0) and end (1.0) of the window.
			Defaults to the center of the window (0.5)
	DELTA_DURATION - (float)   [sec]   Default = 0.001
			Minimum window size in case the start and end of a window are both zero.  Needed so that 
			all windows have a non-zero duration
		OUTPUT
	WindowsOut  - (np_array) [sec]  shape = (Nx2)
			Output of windows built by function.  Values that would have been negative will be rounded to zero.
	'''
	
	#### Valiation of Inputs
	if (TimePoints.shape[0] <= 0):
		print("ERROR: Expected a one dimensional list (first dimension <= 0).")
		raise Exception("ERROR: Expected a one dimensional list (first dimension <= 0).")
	if (len(TimePoints.shape) != 1):
		print("ERROR: Expected a one dimensional list (second dimension != 1).")
		raise Exception("ERROR: Expected a one dimensional list (second dimension != 1).")
	if (any(TimePoints<0)):
		print("ERROR: Expects non-negative Timesteps.")
		raise Exception("ERROR: Expects non-negative Timesteps.")
	if (WindowSz <= 0):
		print("ERROR: Window Size must be greater than 0.")
		raise Exception("ERROR: Window Size must be greater than 0.")
	if (Offset < 0.0   or   Offset > 1.0):
		print("ERROR: Offset must be a value between 0.0 and 1.0 inclusive for the start or end of window.")
		raise Exception("ERROR: Offset must be a value between 0.0 and 1.0 inclusive for the start or end of window.")
	# end error statements
	
	
	#### Convert TimePoints
	WindowsOut = np.zeros([len(TimePoints),2]) # intialize memory
	
	LeadingOffset  = WindowSz*Offset
	TrailingOffset = WindowSz*(1-Offset) 
	
	WindowsOut[:,0] = (TimePoints - LeadingOffset)
	WindowsOut[:,1] = TimePoints + TrailingOffset
	
	# Round negatives to zero
	WindowsOut[WindowsOut < 0.0] = 0.0
	# Check that no start and end time is the same (windows must have a non-zero duration)
	if (Offset==1.0): # Should only tringger if 0 was an input and the window offset is 1.0
		WindowsOut[WindowsOut[:,1] == WindowsOut[:,0],1] += DELTA_DURATION # add delta offset

	
	
		

	#### Return Final Results
	return WindowsOut
# end of TimePoint2Window()







def Window2TimePoint(WindowTimes, Offset = 0.5):
	'''
	Func: TimePoint2Window()
	Purpose: Takes specific points of time (predictions, GT, etc.) and creates a window of time for each of these predictions.
			It allows for tuning the size of the windows made and the placement of window about the event.
		### WARNING ###:
	This function does not check if windows are overlapping once made.
		INPUTS
	WindowTimes - (np_array)   [sec]   shape = (Nx2)
			Input sequence of windows to convert.  Assumed to be in seconds, and non-negative.
	Offset     - (float)   [%]   Default = 0.5, Range = [0.0, 1.0]
			Where to place the event relative to the start (0.0) and end (1.0) of the window.
			Defaults to the center of the window (0.5)
		OUTPUT
	PointsOut  - (np_array) [sec]  shape = (Nx2)
			Output of windows built by function.  Values that would have been negative will be rounded to zero.
	'''
	
	#### Valiation of Inputs
	if (np.shape(WindowTimes)[0] <= 0):
		print("ERROR: Expected at least one element in Windows (first dimension <= 0).")
		raise Exception("ERROR: Expected at least one element in Windows (first dimension <= 0).")
	if (len(np.shape(WindowTimes)) != 2):
		print("ERROR: Expected a two dimensional list (list of timepoint pairings) (second dimension != 2).")
		raise Exception("ERROR: Expected a two dimensional list (list of timepoint pairings) (second dimension != 2).")
	if (any((WindowTimes[:,1] - WindowTimes[:,0]) < 0)):
		print("ERROR: Expected a pairings to be (start, end). (some end-start<0).")
		raise Exception("ERROR: Expected a pairings to be (start, end). (some end-start<0).")
	if ((WindowTimes<0).any()):
		print("ERROR: Expects non-negative start and stop times.")
		raise Exception("ERROR: Expects non-negative start and stop times.")
	if (Offset < 0.0   or   Offset > 1.0):
		print("ERROR: Offset must be a value between 0.0 and 1.0 inclusive for the start or end of window.")
		raise Exception("ERROR: Offset must be a value between 0.0 and 1.0 inclusive for the start or end of window.")
	# end of Validations



	
	#### Convert Windows To TimePoints

	#Compute start minus end times
	WinSizes = WindowTimes[:,1] - WindowTimes[:,0]

	# Scale to Find Offset for each window
	All_Offsets = WinSizes * Offset

	# Create output as timepoint offset from start of window
	PointsOut = WindowTimes[:,0] + All_Offsets


	# Return Final Results
	return PointsOut

# end of Window2TimePoint()






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








#############################################################
###########         Evaluation  FUNCTIONS         ###########
#############################################################



def DongEval_PtVsPt(Predictions_pts, GT_pts):
	'''
	Func: DongEval_PtVsPt()
	Purpose: Takes a single meal worth of data (time point predictions and time point Ground Truth)
			and matches greedily.  Returns the True Positives (TP), False Positives (FP), and False 
			Negatives (FN).
			  Matching scans the ground truths for a match after the previous detection and before 
			the next detection, prioritizing the latest match before current and then the earliest
			match after.
		INPUTS
	Predictions_pts  - 1D numpy array [sec]
			List of time points where a event was predicted to occur 
	GT_pts   - 1D numpy array [sec]
			List of time points where events will be matched
		OUTPUT
	Metrics  - list of [TP, FP, FN]
			TP - int [#] - True positives that match (total GT and Pred pairings)
			FP - int [#] - False positives (Unmatched Detections)
			FN - int [#] - False Negatives (unmatched Ground Truth)
			Key - str  - A String stating what each of the returns are reference
		********* NOTE *********
	Future edits could allow for a verbose output that would return the comp_matched and 
			gt_matched variables.  These additional returns would allow for easy graphing
			of evaluation pairs.
	'''
	
	
	#### Valiation of Inputs
	if (np.shape(Predictions_pts)[0] <= 0):
		print("ERROR: Expected a one dimensional list for Predictions_pts (first dimension <= 0).")
		raise Exception("ERROR: Expected a one dimensional list for Predictions_pts (first dimension <= 0).")
	if (len(Predictions_pts.shape) != 1):
		print("ERROR: Expected a one dimensional list for Predictions_pts (second dimension != 1).")
		raise Exception("ERROR: Expected a one dimensional list for Predictions_pts (second dimension != 1).")
	if ((Predictions_pts<0).any()):
		print("ERROR: Expects non-negative times for Predictions_pts.")
		raise Exception("ERROR: Expects non-negative times for Predictions_pts.")
	if (np.shape(GT_pts)[0] <= 0):
		print("ERROR: Expected a one dimensional list for GT_pts (first dimension <= 0).")
		raise Exception("ERROR: Expected a one dimensional list for GT_pts (first dimension <= 0).")
	if (len(GT_pts.shape) != 1):
		print("ERROR: Expected a one dimensional list for GT_pts (second dimension != 1).")
		raise Exception("ERROR: Expected a one dimensional list for GT_pts (second dimension != 1).")
	if ((GT_pts<0).any()):
		print("ERROR: Expects non-negative times for GT_pts.")
		raise Exception("ERROR: Expects non-negative times for GT_pts.")
	# end of Validations
	
	
	
	
	
	# NEW GREEDY MATCH BASED ON DONG
	#### Initialize Variables

	gt_matched=np.zeros(len(GT_pts)) # logical for if a GT_bite has been matched to a detection
	comp_matched=np.zeros(len(Predictions_pts)) # logical for if a GT_bite has been matched to a detection
	gt_index=0 # rover to move through GT bites as they are passed by the dectections 
	PredictionOutcomes=[] # track which Detections are matched as TP and which are FP 

	TP=0 # Bites that trigger within ground truth
	FP=0 # False Positives
	FN=0 # GT windows without a bite detected

	if len(Predictions_pts) == 1: # Special Case to avoid illegal indexing
			#Prioritize latest point before prediction
			MatchesBefore = GT_pts[GT_pts< Predictions_pts[0]]
			if len(MatchesBefore)!=0:
				comp_matched[0] = np.max(MatchesBefore)
				gt_matched[GT_pts == np.max(MatchesBefore)]=comp_matched[0]
			else: #if no matches, go with earliest bite after detection but before next
				MatchesAfter = GT_pts
				comp_matched[0] = np.min(MatchesAfter)
				gt_matched[GT_pts == np.min(MatchesAfter)]=comp_matched[0]
			# end of if matching before or matching after
		# end of checking first index
	else: # Standard case for at least two predicitions
		# First is a special case
		if any(np.logical_and(np.logical_not(gt_matched),
							  GT_pts < Predictions_pts[1])):
			#Prioritize latest point before prediction
			MatchesBefore = GT_pts[GT_pts< Predictions_pts[0]]
			if len(MatchesBefore)!=0:
				comp_matched[0] = np.max(MatchesBefore)
				gt_matched[GT_pts == np.max(MatchesBefore)]=comp_matched[0]
			else: #if no matches, go with earliest bite after detection but before next
				MatchesAfter = GT_pts[GT_pts < Predictions_pts[1]]
				comp_matched[0] = np.min(MatchesAfter)
				gt_matched[GT_pts == np.min(MatchesAfter)]=comp_matched[0]
			# end of if matching before or matching after
		# end of checking first index


		for i in range(1, len(Predictions_pts)-1): # middle indices are general case
			if any(np.logical_and(np.logical_not(gt_matched),
								  np.logical_and(GT_pts > Predictions_pts[i-1],
												 GT_pts < Predictions_pts[i+1]))):
				#Prioritize latest point before prediction but after previous detection
				MatchesBefore = GT_pts[np.logical_and(np.logical_not(gt_matched),
									   np.logical_and(GT_pts > Predictions_pts[i-1],
													  GT_pts <= Predictions_pts[i]))]
				if len(MatchesBefore)!=0:
					comp_matched[i] = np.max(MatchesBefore)
					gt_matched[GT_pts == np.max(MatchesBefore)]=comp_matched[i]
				else: #if no matches, go with earliest bite after detection but before next
					MatchesAfter = GT_pts[np.logical_and(np.logical_not(gt_matched),
										  np.logical_and(GT_pts > Predictions_pts[i],
										  GT_pts < Predictions_pts[i+1]))]
					comp_matched[i] = np.min(MatchesAfter)
					gt_matched[GT_pts == np.min(MatchesAfter)]=comp_matched[i]
				# end of if matching before or matching after
				# print("MatchFound")
			else:
				continue
				# print("NOT MATCHABLE")
			# end of if
		# End of for i loop


		# Address special case for last index
		if any(np.logical_and(np.logical_not(gt_matched),
							  GT_pts > Predictions_pts[-2])): #check after second to last index
			#Prioritize latest point before prediction but after previous detection
			MatchesBefore = GT_pts[np.logical_and(np.logical_not(gt_matched),
												  GT_pts > Predictions_pts[-2])]
			if len(MatchesBefore)!=0:
				comp_matched[-1] = np.max(MatchesBefore)
				gt_matched[GT_pts == np.max(MatchesBefore)]=comp_matched[-1]
			else: #if no matches, go with earliest bite after detection but before next
				MatchesAfter = GT_pts[np.logical_and(np.logical_not(gt_matched),
									  GT_pts > Predictions_pts[-1])]
				comp_matched[-1] = np.min(MatchesAfter)
				gt_matched[GT_pts == np.min(MatchesAfter)]=comp_matched[-1]
			# end of if matching before or matching after
			# print("MatchFound")
		# end of if






	for i in range(len(gt_matched)):
		if gt_matched[i]!=0:
			TP+=1
		else:
			FN+=1
	for i in range(len(Predictions_pts)):
		if comp_matched[i]!=0:
			PredictionOutcomes.append(0)
		else:
			FP+=1 #arbitrarily choose FP1 since there is only one FP class
			PredictionOutcomes.append(1)
	
	# #Debug to see why New method gets better results than Old
	# if (TP ==  51 and FP == 0 and FN == 8):
	# 	print(comp_matched)
	# 	print(gt_matched)
	# if (TP ==  54 and FP == 3 and FN == 13):
	# 	print(comp_matched)
	# 	print(gt_matched)
	# Collect all results to return
	Key = "RETURN KEY: TP, FP, FN, Key"
	EvalResults = [TP, FP, FN, Key]
	
	# Return Final Results
	return EvalResults
# end of DongEval_PtVsPt()










def KyritEval_PtVsWindow(Predictions_pts, GT_windows, WIN_TOLERANCE = 0.0, BEFORE_TOL = 0.0, AFTER_TOL = 0.0):
	'''
	Func: KyritEval_PtVsWindow()
	Purpose: Edited version of Kyritsis Evaluation, which matches the first detection within a window
			as a TP, additional detections as FP1, triggers outside of windows as FP2, and missed 
			windows as FN.
				My variation allows for leniency for the bites to be outside of the windows, defined by 
			WIN_TOLERANCE.  
				To run strictly the original proposed method, run with WindowTolerance = 0
				
		INPUTS
	Predictions_pts  - 1D numpy array [sec]
			List of time points where a event was predicted to occur
	GT_windows  - (np_array)   [sec]   shape = (Nx2), rows of [start,stop] pairs
			Ground Truth windows of events.  Assumed to be in seconds and non-negative.
	WIN_TOLERANCE - float [sec] Default = 0.0, range = [0.0, +inf)
			Time to Extend Ground Truth to allow leniency in matching.
		OUTPUT
	Metrics  - list of [TP, FP, FN, FP1, FP2, Key]
			TP  - int [#] - True positives that match (total GT and Pred pairings)
			FP  - int [#] - False positives (Unmatched Detections)
			FN  - int [#] - False Negatives (unmatched Ground Truth)
			FP1 - int [#] - False positives that fell within an already claimed GT (double detection)
			FP2 - int [#] - False positives that fell outside of any GT window (false detection)
			Key - str  - A String stating what each of the returns are reference
		********* NOTE *********
	If using a window tolerance, GT windows could be overlapping and pairings may shuffle around, 
	such as an earlier prediction may match with a later window, thus unmatching the later prediction 
	that is within the window proper.  This should not matter since the total number of FPs and TPs 
	should remain the same.  
		********* NOTE *********
	Future edits could allow for a verbose output that would return the comp_matched and 
			gt_matched variables, as well as FP1 and FP2 (instead of sum).  These additional 
			returns would allow for easy graphing of evaluation pairs.
	'''
	#### Valiation of Input
	# Prediction_pts
	if (np.shape(Predictions_pts)[0] <= 0):
		print("ERROR: Expected a one dimensional list for Predictions_pts (first dimension <= 0).")
		raise Exception("ERROR: Expected a one dimensional list for Predictions_pts (first dimension <= 0).")
	if (len(Predictions_pts.shape) != 1):
		print("ERROR: Expected a one dimensional list for Predictions_pts (second dimension != 1).")
		raise Exception("ERROR: Expected a one dimensional list for Predictions_pts (second dimension != 1).")
	if ((Predictions_pts<0).any()):
		print("ERROR: Expects non-negative times for Predictions_pts.")
		print(Predictions_pts)
		raise Exception("ERROR: Expects non-negative times for Predictions_pts.")
	# GT_window
	if (np.shape(GT_windows)[0] <= 0):
		print("ERROR: Expected at least one element in Ground Truths (first dimension <= 0).")
		raise Exception("ERROR: Expected at least one element in Ground Truths (first dimension <= 0).")
	if (len(np.shape(GT_windows)) != 2):
		print("ERROR: Expected a two dimensional list (list of timepoint pairings) (second dimension != 2).")
		raise Exception("ERROR: Expected a two dimensional list (list of timepoint pairings) (second dimension != 2).")
	if (any((GT_windows[:,1] - GT_windows[:,0]) < 0)):
		print("ERROR: Expected a pairings to be (start, end). (some end-start<0).")
		raise Exception("ERROR: Expected a pairings to be (start, end). (some end-start<0).")
	if ((GT_windows<0).any()):
		print("ERROR: Expects non-negative start and stop times.")
		raise Exception("ERROR: Expects non-negative start and stop times.")
	# WIN_TOLERANCE
	if (WIN_TOLERANCE < 0.0):
		print("ERROR: WIN_TOLERANCE must be a non-negative value.")
		raise Exception("ERROR: WIN_TOLERANCE must be a non-negative value.")
	if (BEFORE_TOL < 0.0):
		print("ERROR: BEFORE_TOL must be a non-negative value.")
		raise Exception("ERROR: BEFORE_TOL must be a non-negative value.")
	if (AFTER_TOL < 0.0):
		print("ERROR: AFTER_TOL must be a non-negative value.")
		raise Exception("ERROR: AFTER_TOL must be a non-negative value.")
	# end of Validations
	
	# if Window Tolerance is specified but the Before and After Tolerances are default
	if (WIN_TOLERANCE != 0 and (BEFORE_TOL == 0.0 and AFTER_TOL == 0.0)):
		# Assign the values of Window Tolerance to Both
		BEFORE_TOL = WIN_TOLERANCE
		AFTER_TOL = WIN_TOLERANCE
	# end of default window tolerance
	
	
	

	# Intialize Metrics Counts
	TP=0
	FP1=0
	FP2=0
	FN=0

	gt_matched=np.zeros(len(GT_windows)) # logical for if a GT_bite has been matched to a detection
	gt_index=0 # rover to move through GT bites as they are passed by the dectections 
	DetectionOutcomes=[] # track which Detections are matched as TP and which are FP 

	# Initialize arrays with -1 for not matched and non-negative index of corresponding list for matched
	GT_MatchFlag = np.array([-1]*(len(GT_windows)))
	Pred_MatchFlag = np.array([-1]*(len(Predictions_pts)))

	for i in range(len(Predictions_pts)):
		currPred=Predictions_pts[i]
		### Search all GT_windows to see if the detection falls into one
		WithinWindowCheck = np.logical_and(GT_windows[:,0] - BEFORE_TOL < currPred,
										   currPred < GT_windows[:,1]+AFTER_TOL)

		# If within Window, determine if it is a TP or a FP1
		if (any(WithinWindowCheck)):
			# if the prediction falls within an unmatched GT, mark as TP
			if (any(np.logical_and(WithinWindowCheck, GT_MatchFlag == -1))):
				# grab the earliest window if multiple exist
				PossibleMatches = np.logical_and(WithinWindowCheck, GT_MatchFlag == -1)
				gt_ind = np.min(np.where(PossibleMatches==True)[0])

				TP+=1
				GT_MatchFlag[gt_ind]  = i
				Pred_MatchFlag[i] = gt_ind
				DetectionOutcomes.append(0)
			else: # if all windows are already matched, mark as FP1
				FP1+=1
				DetectionOutcomes.append(1)

		else: # If outside of all windows
			FP2+=1
			DetectionOutcomes.append(2)
		# end of if (Within Any Window)

	# end of for i in Detections loop

	### Check to see how many missed GT events
	for i in GT_MatchFlag:
		if i == -1:
			FN+=1
	# end of FN counting loop
	
	
	
	
	
	# Collect all results to return
	Key = "RETURN KEY: TP, FP, FN, FP1, FP2, Key"
	EvalResults = [TP, FP1+FP2, FN, FP1, FP2, Key]
	
	# Return Final Results
	return EvalResults

# end of KyritEval_PtVsWindow()











def Duration_WindowVsWindow(Pred_windows, GT_windows, START_TIME = -1.0, END_TIME = -1.0):
	'''
	Func: Duration_WindowVsWindow()
	Purpose:  Implementation of the Duration based window to window evaluation proposed by Dong (used 
			in Wei Paper).  Computes the total overlap of all GT and Predictions, treating both as 
			windows and ignoring the stride of the sliding window.  Ideally, teh full GT window is matched
			by a full Prediction window.
				Timeline will be continual, switching between True and False at each GT boundary and 
			flipping Positive and Negative at each Prediction boundary.  Segmenting data into 
			these divisions and matching which of the 4 categories each segment (or the midpoint 
			of each segment since segments must all be the same class) results in total calculation
		INPUTS
	Pred_windows  - (np_array)   [sec]   shape = (Nx2), rows of [start,stop] pairs
			Prediction windows of events.  Assumed to be in seconds and non-negative.
	GT_windows  - (np_array)   [sec]   shape = (Nx2), rows of [start,stop] pairs
			Ground Truth windows of events.  Assumed to be in seconds and non-negative.
	START_TIME  - float   [sec]   default = -1.0, range [0.0,+Inf)
			Time to begin evaluation.  Must be greater than zero if defined. Default value will
			begin at the minimum of the prediction and GT boundaries.
	END_TIME  - float   [sec]   default = -1.0, range [0.0,+Inf)
			Time to stop evaluation.  Must be greater than zero if defined. Default value will
			begin at the minimum of the prediction and GT boundaries.
		OUTPUT
	Metrics  - list of [TP, FP, FN]
			TP - int [#] - True positives that match (total GT and Pred pairings)
			FP - int [#] - False positives (Unmatched Detections)
			FN - int [#] - False Negatives (unmatched Ground Truth)
			Key - str  - A String stating what each of the returns are reference
	'''

	# Eventually validate
	# Pred_window
	if (np.shape(Pred_windows)[0] <= 0):
		print("ERROR: Expected at least one element in Preds (first dimension <= 0).")
		raise Exception("ERROR: Expected at least one element in Preds (first dimension <= 0).")
	if (len(np.shape(Pred_windows)) != 2):
		print("ERROR: Expected a two dimensional list (list of Pred timepoint pairings) (second dimension != 2).")
		raise Exception("ERROR: Expected a two dimensional list (list of Pred timepoint pairings) (second dimension != 2).")
	if (any((Pred_windows[:,1] - Pred_windows[:,0]) < 0)):
		print("ERROR: Expected a Pred pairings to be (start, end). (some end-start<0).")
		raise Exception("ERROR: Expected a Pred pairings to be (start, end). (some end-start<0).")
	if ((Pred_windows<0).any()):
		print("ERROR: Expects non-negative start and stop times.")
		raise Exception("ERROR: Expects non-negative start and stop times.")
	# GT_window
	if (np.shape(GT_windows)[0] <= 0):
		print("ERROR: Expected at least one element in Ground Truths (first dimension <= 0).")
		raise Exception("ERROR: Expected at least one element in Ground Truths (first dimension <= 0).")
	if (len(np.shape(GT_windows)) != 2):
		print("ERROR: Expected a two dimensional list (list of timepoint pairings) (second dimension != 2).")
		raise Exception("ERROR: Expected a two dimensional list (list of timepoint pairings) (second dimension != 2).")
	if (any((GT_windows[:,1] - GT_windows[:,0]) < 0)):
		print("ERROR: Expected a pairings to be (start, end). (some end-start<0).")
		raise Exception("ERROR: Expected a pairings to be (start, end). (some end-start<0).")
	if ((GT_windows<0).any()):
		print("ERROR: Expects non-negative start and stop times.")
		raise Exception("ERROR: Expects non-negative start and stop times.")
	# START_TIME
	if (START_TIME != -1.0  and START_TIME < 0.0):
		print("ERROR: START_TIME must be a non-negative value, or default of -1.0 (starts at minimum value of detections or GT).")
		raise Exception("ERROR: START_TIME must be a non-negative value, or default of -1.0 (starts at minimum value of detections or GT).")
	# END_TIME
	if (END_TIME != -1.0  and END_TIME < 0.0):
		print("ERROR: END_TIME must be a non-negative value, or default of -1.0 (starts at minimum value of detections or GT).")
		raise Exception("ERROR: END_TIME must be a non-negative value, or default of -1.0 (starts at minimum value of detections or GT).")
	# end of validation
	
	
	# Timeline will be continual, switching between True and False at each GT boundary and 
	#   flipping Positive and Negative at each Prediction boundary.  Segmenting data into 
	#   these divisions and matching which of the 4 categories each segment (or the midpoint 
	#   of each segment since segments must all be the same class) results in total calculation
	
	
	# Create partitions
	All_Transitions = np.append(GT_windows.flatten(), Pred_windows.flatten())
	
	# Sort all into one timeline, and combine duplicate (perfectly matching) window boundaries
	Timeline_Transitions= np.unique(np.sort(All_Transitions))
	
	# Midpoints = start + (end-start)/2 = (start+end)/2
	Midpoints = (Timeline_Transitions[1:] + Timeline_Transitions[0:-1]) / 2
	
	
	#Calulate start and end times if needed
	if (START_TIME == -1.0):
		start_time = Timeline_Transitions[0] #default to first timepoint
	else:
		start_time = START_TIME
	#end if
	
	if (END_TIME == -1.0):
		end_time = Timeline_Transitions[-1] #default to latest timepoint
	else:
		end_time = END_TIME
	#end if
	
	
	# Initialize Scoring Metrics
	TP = 0.0
	FP = 0.0
	TN = 0.0
	FN = 0.0
	
	# All Time before first boundary is a True Negative since no predictions or GT exist
	TN += Timeline_Transitions[0] - start_time
	
	# labels=[] # labels used for Graphing, and found in PG_EvalFuncs.ipynb file
	# Removed from EvaluationMethodsFuncs to clean up space when running
	# Also removed DEBUG_PRINTS 


	# All other points can be computed by identifying where the midpoint lies
	for i in range(len(Midpoints)):
		currMidpoint=Midpoints[i]

		WithinPreds = any(np.logical_and(Pred_windows[:,0] < currMidpoint,
										 currMidpoint < Pred_windows[:,1]))
		WithinGT = any(np.logical_and(GT_windows[:,0] < currMidpoint,
									  currMidpoint < GT_windows[:,1]))
		
		# Amount of time to add
		WindowDuration = Timeline_Transitions[i+1] - Timeline_Transitions[i]
		# Add to respective Classes
		if (WithinPreds and WithinGT):
			TP += WindowDuration
		elif (WithinPreds and (not WithinGT)):
			FP += WindowDuration
		elif ((not WithinPreds) and WithinGT):
			FN += WindowDuration
		elif ((not WithinPreds) and (not WithinGT)):
			TN += WindowDuration
		else:
			raise Exception("Should not have gotten to this transition")
		# end of which class window segment falls into	
	# end of for i in Midpoints
	
	# All Time after last boundary is also True Negative since no predictions or GT exist
	TN += end_time - Timeline_Transitions[-1]
	
	
	# Collect all results to return
	Key = "RETURN KEY: TP, FP, FN, TN, Key"
	EvalResults = [TP, FP, FN, TN, Key]
	
	# Return Final Results
	return EvalResults

# end of Duration_WindowVsWindow()











def Section_WindowVsWindow(Inputs):
	'''
	Func: Section_WindowVsWindow()
	Purpose: Description
		INPUTS
	Inputs   - Description
		OUTPUT
	Outputs  - Description
	'''
	
	
	# Return Final Results
	return Output
# end of Section_WindowVsWindow()













#####################################################
###########       Printing FUNCTION       ###########
#####################################################


def PrintStats(TP, FP, FN, FP1 = -1, FP2 = -1, VerbosePrintOpt = 0):
	'''
	Func: FuncName()
	Purpose: Takes evaluation results (i.e. True Pos, False Pos, False Neg, etc) and computes 
			evaluation metrics (e.g. F1 Score), then prints out results.
		INPUTS
	Inputs   - Description
		OUTPUT
	Outputs  - Description
	'''
	
	
	# Compute Precision (TPR) Recall (PPV), and F1 Score 
	if (TP+FN)!=0:
		TPR = (float(TP) / (float(TP+FN)))*100
	else:
		TPR = -1 # mark as an Error
		
	if (TP+FP)!=0:
		PPV = (float(TP) / (float(TP+FP)))*100
	else:
		PPV = -1 # mark as an Error
	
	if (TPR+PPV)!=0:
		F1_score = 2*TPR*PPV / (TPR+PPV)
	else:
		F1_score = -1 # mark as an Error
	
	
	if VerbosePrintOpt == 0: # Basic Print
		print('TP={0:0d} FP={1:0d} U={2:0d}'.format(TP,FP,FN))
		if (FP1 != -1 and FP2!=-1):
			print('\tFP1={0:0d} FP2={1:0d}'.format(FP1,FP2))
		print('TPR={0:.3f}'.format(TPR))
		print('PPV={0:.3f}'.format(PPV))
		print('F1_Score={0:.3f}'.format(F1_score))

		# if PLACEMENT_METHOD_FLAG == 1:
		# 	print('SW Placement=Offset, Offset = {} sec'.format(PLACEMENT_METHOD_1_OFFSET))
		# elif PLACEMENT_METHOD_FLAG == 2:
		# 	print('SW Placement=Midpoint')
		# #end of PLACEMENT_METHOD_FLAG switch

	else: # Single Line Print for Mass Testing
		if PLACEMENT_METHOD_FLAG == 1:
			print('{0:.3f} {1:.3f} {2:.3f} {3:0d} {4:0d} {5:0d} {6} {7:0d} Eval=Dong Place=Off Offset={8:.2f}'.format(
				F1_score,  TPR, PPV, TP, FP, FN, sys.argv[1], FOLD_INDEX, PLACEMENT_METHOD_1_OFFSET))
		elif PLACEMENT_METHOD_FLAG == 2:
			print('{0:.3f} {1:.3f} {2:.3f} {3:0d} {4:0d} {5:0d} {6} {7:0d} Eval=Dong Place=Midpt'.format(
				F1_score,  TPR, PPV, TP, FP, FN, sys.argv[1], FOLD_INDEX))
		else:
			print("ERROR: NO PLACEMENT METHOD PROPERLY SELECTED; SET EVAL_METHOD_FLAG TO ACCEPTABLE VALUE.")


	
	
	# Return
	return
# end of PrintStats()




def PrintStats_SingleLine(TP, FP, FN, FP1 = -1, FP2 = -1, VerbosePrintOpt = 0):
	'''
	Func: FuncName()
	Purpose: Takes evaluation results (i.e. True Pos, False Pos, False Neg, etc) and computes 
			evaluation metrics (e.g. F1 Score), then prints out results.
		INPUTS
	Inputs   - Description
		OUTPUT
	Outputs  - Description
	'''
	
	# Compute Precision (TPR) Recall (PPV), and F1 Score 
	if (TP+FN)!=0:
		TPR = (float(TP) / (float(TP+FN)))*100
	else:
		TPR = -1 # mark as an Error
	# end of TPR Calc
	
	if (TP+FP)!=0:
		PPV = (float(TP) / (float(TP+FP)))*100
	else:
		PPV = -1 # mark as an Error
	# End of PPV Calc
	
	if (TPR+PPV)!=0:
		F1_score = 2*TPR*PPV / (TPR+PPV)
	else:
		F1_score = -1 # mark as an Error
	# end of F1 Calc
	
	if VerbosePrintOpt == 1:
		print('{0:6s} {1:6s} {2:6s} {3:6s} {4:6s} {5:6s}'.format(
			"F1",  "TPR", "PPV", "TP", "FP", "FN"))
	# end of verbose header
	
	# Print Stats
	print('{0:.3f} {1:.3f} {2:.3f} {3:<6d} {4:<6d} {5:<6d}'.format(
		F1_score, TPR, PPV, TP, FP, FN))
	
	# Return
	return 

# end def PrintStats_SingleLine()

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





