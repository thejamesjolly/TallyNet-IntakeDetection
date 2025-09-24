'''
File: DongHeur_Benchmarks.py
Author: James Jolly
Purpose: 
	Evaluation of Dong Roll Heuristic Intake detection benchmark on all one handed datasets
	Turn input parameters to produce desired results.
	
	Results in Paper were tuned to produce balanced scores are provided below.
	Inputs are given in the following order:
		[Dataset] [roll_thresh] [time2_thresh_sec]

	DND - One Hand OREBA
		1 10 8
	DND - Clemson
		2 10 6
	Dom - One Hand OREBA
		3 25 6
	Dom - Clemson
		4 10 8
'''




####################
##     Imports    ##
####################

# Notebook to Read OREBA and ClemCafe Data to run Heuristic Approach and compute F1, TPR, and PPV


import numpy as np
import time
import sys
from io import UnsupportedOperation
import pickle as pkl
import csv


sys.path.insert(1, './../AuxillaryFunctions')

# import personal Functions
from GenerateClassDataFuncs import ResampleData
from EvaluationMethodFuncs import TimePoint2Window
from EvaluationMethodFuncs import KyritEval_PtVsWindow
from EvaluationMethodFuncs import DongEval_PtVsPt
from EvaluationMethodFuncs import PrintStats
from EvaluationMethodFuncs import PrintStats_SingleLine





#####################
##   Definitions   ##
#####################






#####################
##   Main Func()   ##
#####################

if __name__ == '__main__':
	
	
	### USER INPUT ###
	Database_Flag = int(sys.argv[1])
	# 1 = OREBA (DND GT)
	# 2 = Clemson (DND GT)
	# 3 = OREBA (Dom GT Only)
	# 4 =  Clemson (Dom GT Only)

	roll_thresh = float(sys.argv[2]) #	units are deg/sec) 

	time2_thresh_sec = float(sys.argv[3])
	
	# Outputs all detections (Boolean)
	SAVE_DETECTIONS_TO_FILES = True



	### DEFINE CONSTANTS ###

	Database_Names = ['None Selected','OREBA (One Hand)', 'Clemson', 'OREBA One Hand (Dom GT Only)', 'Clemson(Dom GT Only)']
	Database_Acronym = ['', 'DndOHO', 'DndClemson', 'DomOHO', 'DomClemson']

	if Database_Flag == 1: # OREBA
		DATABASE_FILEPATH = "./../Pickle_Databases/Raw_OneHandOreba.pkl"
		OrigFreq = 64 # [Hz] Oreba data collection rate
		# time2_thresh = 9*OrigFreq	(8 sec)
	elif Database_Flag == 2: # Clemson
		DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_Clemson.pkl"
		OrigFreq = 15 # [Hz] Oreba data collection rate
		# time2_thresh = 6*OrigFreq	(6 sec) # Updated value from Shen
	elif Database_Flag == 3: # OREBA
		GT_DATABASE_FILEPATH = "./../Pickle_Databases/Dom_OneHandOreba.pkl"
		DATABASE_FILEPATH = "./../Pickle_Databases/Raw_OneHandOreba.pkl"
		OrigFreq = 64 # [Hz] Oreba data collection rate
		# time2_thresh = 6*OrigFreq	(8 sec)
	elif Database_Flag == 4: # Clemson
		DATABASE_FILEPATH = "./../Pickle_Databases/Dom_Clemson.pkl"
		OrigFreq = 15 # [Hz] Oreba data collection rate
		# time2_thresh = 8*OrigFreq	(6 sec) # Updated value from Shen
	# end of database switch

	time1_thresh = 2*OrigFreq	# units are 15Hz intervals (2 sec) */
	time2_thresh = int(round(time2_thresh_sec * OrigFreq)) # convert to data units



	######################
	### Grab Roll Data ###
	######################

	# SMOOTHING = 7 #number of points on either side of a point to average
	SMOOTHING = int(np.floor((OrigFreq-1)/2)) # about one second of total smoothing for OREBA
	RESAMPLE_FLAG = 0 # Freq [Hz] to format final data


	with open(DATABASE_FILEPATH,'rb') as fh:
		dataset = pkl.load(fh)
	# OREBA_Cucumber={'UniqueID','raw_data','handedness','bites_gt'}
	# ClemCafe_Cucumber={README,UniqueID,proc_data_rawAcc,proc_data_noG,handedness, ...
	#             ... bites_gt, bites_handedness_gt, means_global_rawAcc, ...
	#             ... stddev_global_rawAcc, means_global_noG, stddev_global_noG}

	# Grab data from Database and define constants for 
	if Database_Flag == 1: # if OREBA
		RollCol=4
		raw = dataset['raw_domHand_data']
		rawRoll = []
		for meal_idx in range(len(raw)):
			rawRoll.append(raw[meal_idx][:,RollCol]) # Just Grab roll
		# end of for i in len(raw)

		biteGT_windows=dataset['bites_gt']
		biteGT_times= []
		for i in range(len(biteGT_windows)):
			# place time in the middle of the window
			biteGT_times.append(biteGT_windows[i][:,0] + ((biteGT_windows[i][:,1]-biteGT_windows[i][:,0])/2))
		# end of for i

	elif Database_Flag == 3: # if OREBA Dom GT Only
		# Grab Dom limited GT OHO Data 
		with open(GT_DATABASE_FILEPATH,'rb') as fh:
			dataset_GT = pkl.load(fh)

		RollCol=4
		raw = dataset['raw_domHand_data']
		rawRoll = []
		for meal_idx in range(len(raw)):
			rawRoll.append(raw[meal_idx][:,RollCol]) # Just Grab roll
		# end of for i in len(raw)

		biteGT_windows=dataset_GT['bites_gt']
		biteGT_times= []
		for i in range(len(biteGT_windows)):
			# place time in the middle of the window
			biteGT_times.append(biteGT_windows[i][:,0] + ((biteGT_windows[i][:,1]-biteGT_windows[i][:,0])/2))
		# end of for i


	elif Database_Flag == 2 or Database_Flag == 4: # if Clemson
		RollCol=5
		normalized_motion = dataset['proc_data_rawAcc']
		rawRoll = []
		for meal_idx in range(len(normalized_motion)):
			normRollData = normalized_motion[meal_idx][:,RollCol]
			# Un-normalize the Data
			rawMealRoll = np.zeros(np.shape(normRollData))
			# For Clemson Data
			MeansGlobal = dataset['means_global_rawAcc']
			StdDevGlobal = dataset['stddev_global_rawAcc']
			# MeansGlobal = np.array([5.67246264e-01, -2.89351987e-01, 6.06367364e-01, 1.07032646e-15, 2.00383723e-15, -3.91199091e-16])
			# StdDevGlobal = np.array([0.43807806, 0.32371267, 0.38916464, 18.0970793, 18.85461884, 37.44610756])

			rawMealRoll[:] = (normRollData[:]*StdDevGlobal[RollCol])+MeansGlobal[RollCol] # Zscore=(value-mean)/StdDev

			rawRoll.append(rawMealRoll)
		# end of for i

		biteGT_indices=dataset['bites_gt']
		biteGT_times = []
		biteGT_windows = []
		for i in range(len(biteGT_indices)):
			biteGT_times.append(biteGT_indices[i]/OrigFreq) #convert to time domain to maintain data after resampling
			biteGT_windows.append(TimePoint2Window(biteGT_times[i], WindowSz=6.5,Offset=0.15))
		# end of for i in GT_indices
	# end of if Database_Flag



	totalMeals = len(rawRoll)



	######################################################
	### RUN DONG HEURISTIC METHOD TO CREATE DETECTIONS ###
	######################################################

	if SMOOTHING != 0:
		print("SMOOTHING DATA BY REACH OF {} DATA OR {:.2f} SECONDS".format(SMOOTHING, SMOOTHING/OrigFreq))
	
	
	meals_for_fold = range(len(rawRoll)) #Full Set

	UniqueIDs = dataset['UniqueID'] # used for Detection Output
	FivePercentCheckpoint = round(len(meals_for_fold)/20)
	Detections=[]

	for meal_num in meals_for_fold:

		#Grab current meal gt bites and data
		OG_meal_data=rawRoll[meal_num]

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

		### SMOOTH FOR DONG HUERISTIC METHOD
		if SMOOTHING != 0:
			totalData = len(meal_data)

			SmoothedData=np.zeros(np.shape(meal_data))
			for i in range(totalData):
				SmoothedData[i]=meal_data[i]
			if False:
				for i in range(SMOOTHING, totalData - SMOOTHING):
					total = 0.0
					for k in range(i-SMOOTHING, i+SMOOTHING+1):
						if (k >= 0 and k < totalData):
							total += meal_data[k]
					SmoothedData[i] = total / (SMOOTHING*2 + 1)
					# end of for j loop
				# end of for i loop
			if True: #smooth previous 14 points
				for i in range(SMOOTHING*2, totalData):
					total = 0.0
					for k in range(i-2*SMOOTHING, i+1):
						if (k >= 0 and k < totalData):
							total += meal_data[k]
					SmoothedData[i] = total / (SMOOTHING*2 + 1)
				# end of for i loop
			meal_data=SmoothedData
		### END OF SMOOTHING ###

		#Grab minimum and maximum for ground truth
		start = min(biteGT_times[meal_num]) - 10 # start 10 seconds back from earliest bite
		start = max(start,0) # safeguard overreaching
		end   = max(biteGT_times[meal_num]) + 10 # end 10 seconds after latest bite
		end   = min(end,len(meal_data)/OrigFreq-1) # safeguard overreaching

		roll=meal_data # pull out just the roll data [deg/sec]
		roll_avg = np.mean(roll) #average the value for zero reference

		# IMPLEMENT CODE FOR LOOP
		bitecount=0
		event=0

		curr_detections=[]

		for i in range(int(start)*OrigFreq,int(end)*OrigFreq):
			if (event == 0  and  roll[i] > (roll_avg+roll_thresh)):
				event=1;
				dt=0;
			if (event == 1):
				dt+=1
				if (dt > time1_thresh):	#/* count up to 2 seconds */
					event=2;
			if (event == 2  and  roll[i] < (roll_avg-roll_thresh)):
				event=3
				dt=0
				bitecount+=1
				# print("{:0.2f}".format(i/OrigFreq))	#/* index of bite detected */
				curr_detections.append(i/OrigFreq)
			if (event == 3):
				dt+=1
				if (dt > time2_thresh):	#/* count up to 8 seconds */
					event=0
			if False: #DEBUG:
				print("index {:d}  value {:f}  event {:d}\n".format(i/OrigFreq,roll[i],event))
		# end of i loop

		if (meal_num%FivePercentCheckpoint) == 0:
			print("Done through Meal {} ({:0.0f}%)".format(meal_num, 100.0*(meal_num+1)/len(meals_for_fold)))
		Detections.append(curr_detections)
		
		
		if SAVE_DETECTIONS_TO_FILES:
			currID = UniqueIDs[meal_num]

			filename = "./Detections/{}/Dets_{}_{}.txt".format(Database_Acronym[Database_Flag], Database_Acronym[Database_Flag], currID)
			with open(filename,"w") as f:
				Header = "Dataset = {}; RollThresh [deg/sec] = {}; Time2Thresh [sec] = {}; Detections given in seconds from start of data\n".format(
					Database_Names[Database_Flag],
					roll_thresh,
					time2_thresh_sec
				)
				f.write(Header)

				TextDetections=[]
				for currDetection in curr_detections:
					TextDetections.append("{:0.2f}".format(currDetection)) # save all detections to 2 decimals
				# end for
				FullDetectionText = '\n'.join(TextDetections)+'\n'
				f.write(FullDetectionText)
			# end of with open()
		# end of if SAVE_DETECTIONS_TO_FILES
		
		
		
	# end of meal_num loop

	print("Finished Making Detections with Dong Heuristics")
	
	
	





	#####################
	### Pt to Pt Eval ###
	#####################

	# Detections already made
	All_TP=[] # List (one entry per file) of bites that trigger within ground truth
	All_FP=[] # List of Bites that trigger but have already been mapped to a TP
	All_FN=[] # List of GT windows without a bite detected


	for currEvalIdx in range(len(meals_for_fold)):
		currDetections = np.array(Detections[currEvalIdx])
		currBiteGT_times = np.array(biteGT_times[currEvalIdx])

		if len(currDetections) == 0:
			# if no detections, then all GT are false positives
			TP = 0
			FP = 0
			FN = len(currBiteGT_times)
			# print("NO DETECTIONS FOR MEAL INDEX {}".format(currEvalIdx)) # Optional Print
		else:
			[TP, FP, FN, Key] = DongEval_PtVsPt(currDetections, currBiteGT_times)
		#end of EVAL_METHOD_FLAG Switch

		All_TP.append(TP)
		All_FP.append(FP)
		All_FN.append(FN)
	#end for currEvalIndex


	Final_TP = sum(All_TP) # overwrite the short names with the total results
	Final_FP = sum(All_FP)
	Final_FN = sum(All_FN)

	print("Dataset={}, Eval = DongPt2Pt".format(Database_Names[Database_Flag]))
	PrintStats_SingleLine(Final_TP,Final_FP,Final_FN, VerbosePrintOpt = 1)



	#########################
	### Pt to Window Eval ###
	#########################

	# Detections already made
	All_TP=[] # List (one entry per file) of bites that trigger within ground truth
	All_FP=[] # List of Bites that trigger but have already been mapped to a TP
	All_FN=[] # List of GT windows without a bite detected

	WIN_TOLERANCE = 8

	for currEvalIdx in range(len(meals_for_fold)):
		currDetections = np.array(Detections[currEvalIdx])
		currBiteGT_windows = np.array(biteGT_windows[currEvalIdx])

		if len(currDetections) == 0:
			# if no detections, then all GT are false positives
			TP = 0
			FP = 0
			FN = np.shape(currBiteGT_windows)[1]
			# print("NO DETECTIONS FOR MEAL INDEX {}".format(currEvalIdx)) # Optional Print
		else:
			[TP, FP, FN, FP1, FP2, Key] = KyritEval_PtVsWindow(currDetections, currBiteGT_windows, WIN_TOLERANCE = WIN_TOLERANCE)
		#end of EVAL_METHOD_FLAG Switch

		All_TP.append(TP)
		All_FP.append(FP)
		All_FN.append(FN)

	#end for currEvalIndex

	Final_TP = sum(All_TP) # overwrite the short names with the total results
	Final_FP = sum(All_FP)
	Final_FN = sum(All_FN)

	print("Dataset={}, Eval = KyritTol, Tol = {}".format(Database_Names[Database_Flag], WIN_TOLERANCE))
	PrintStats_SingleLine(Final_TP,Final_FP,Final_FN, VerbosePrintOpt = 1)



# end of main()
