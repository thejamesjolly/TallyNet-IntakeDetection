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
from GenerateClassDataFuncs import GenerateEvalData_ClemCafe
from GenerateClassDataFuncs import GenerateEvalData_OREBA

from EvaluationMethodFuncs import TimePoint2Window
from EvaluationMethodFuncs import Window2TimePoint
from EvaluationMethodFuncs import DongEval_PtVsPt
from EvaluationMethodFuncs import KyritEval_PtVsWindow
from EvaluationMethodFuncs import Duration_WindowVsWindow
from EvaluationMethodFuncs import PrintStats

from DetectionPlacementFuncs import Cnt2Detect_MidpointTriggerMidpointPlacement
from DetectionPlacementFuncs import Cnt2Detect_TetrisBlockFilling
from DetectionPlacementFuncs import Cnt2Detect_TemplateMatching
from DetectionPlacementFuncs import Cnt2Detect_TetrisBlockFilling_Midpt


if __name__ == '__main__':
	
	DEBUG_PRINTS = 0
		# Set to 1 for additional Prints throughout the file,
		# clear to 0 for only final output
	VERBOSE_PRINT = 0
		# set to 1 for Lengthy Output, 0 for single line
	TIMING_PRINT = 0 
		# Set to 1 if prints for times should be made, 0 if not desired
	
	################################
	###  INPUT VARIABLE PARSING  ###
	################################
	
	NUM_REQUIRED_INPUTS = 6
	##### define required inputs for the data #####
	MODEL_NAME = sys.argv[1]
	DATABASE_FLAG = int(sys.argv[2])
	FOLDS_TOTAL = int(sys.argv[4])
	FOLD_INDEX = int(sys.argv[3])%FOLDS_TOTAL # wrap data into valid fold number in case fold 
					# index is given as [1,FOLDS_TOTAL] instead of [0,FOLDS_TOTAL) 
	CUT_sec = float(sys.argv[5]) # the total length of the window in [sec]

	##### Optional arguments with default values #####
	STRIDE_sec = 1 # The time to move between each window when making examples in [sec]
	RESAMPLE_FLAG_FREQ = 15 # zero if keeping original data frequency, new freq otherwise
	RR_FLAG = 1 # 1 if using striped fold division, 0 if using block fold division
	SAVE_PREDICITONS_FLAG = 0 # Saves the predictions to a CSV
	SAVE_DETECTIONS_FLAG = 0 # Saves the Detections to a CSV
	SAVE_GT_FLAG = 0 # Saves the GT to a CSV
	save_preds_filename = 'dummyfile.txt'
	save_dets_filename = 'dummyfile.txt'
	save_gts_filename = 'dummyfile.txt'

	# Detection Variables
	DetectionPlacementFlag = 1
		# 1 = Midpoint Trigger, Midpoint Placement
		# 2 = Tetris Block Filling and trigger
		# 3 = Template Matching Search
		# 4 = Tetris Block v2: Trigger at Midpoint
	DETECTION_METHOD_NAMES = ["Names for Eval methods (valid flags start at 1)",
						 "Midpoint Trigger",
						 "Tetris v1",
						 "TemplateMatching",
						 "Tetris v2"
						]
	MATCH_THRESH = 0.8 # score (as a percent) required to match with template

	# Evaluation Variables
	EVAL_METHOD_FLAG = 2
		# 1 = Dong Eval (Pt vs Pt)
		# 2 = Kyritsis (Pt vs Window)
		# 3 = Duration (Window vs Window)
	EVAL_METHOD_NAMES = ["Names for Eval methods (valid flags start at 1)",
						 "Dong_PtVsPt",
						 "Kyritsis_PtVsWind",
						 "Duration_WindVsWind"]
	WIN_TOLERANCE = 0.0 # If using Kyritsis_PtVsWind
		# default to 0.0 for strict method, otherwise provide [sec] preds can be away from window

	##### Unused but could be used for data generation inputs #####
	DS_FLAG = 2 # 1 = linear interp, 2 = Cubic Interp
	DS_Rate = 1 # how many data points to collapse and downsample the data (e.g. 4 -> quarters Freq)
	DS_Delay = 0 # what offset to use from first data when downsampling (up to DS_Rate)
	SMOOTHING_FACTOR = 0 # the number of data points to either side to smooth the data
	SCALE_FACTOR_FOR_BITE_COUNTS = 0 # Used if Scaling the model predictions.
		# Set to 0 if not using, and any other value if so

	if len(sys.argv) > NUM_REQUIRED_INPUTS:
		i = NUM_REQUIRED_INPUTS # use while loop to modify index rather than for loop with range 
		while i < len(sys.argv):
			opt_arg = sys.argv[i]
			if opt_arg == '--stride':
				STRIDE_sec = float(sys.argv[i+1])
			elif opt_arg == '--resamp':
				RESAMPLE_FLAG_FREQ = int(sys.argv[i+1])
			elif opt_arg == '--fold-seg':
				RR_FLAG = int(sys.argv[i+1])
			elif opt_arg == '--save-preds':
				SAVE_PREDICITONS_FLAG = 1
				save_preds_filename = (sys.argv[i+1])
			elif opt_arg == '--save-dets':
				SAVE_DETECTIONS_FLAG = 1
				save_dets_filename = (sys.argv[i+1])
			elif opt_arg == '--save-gts':
				SAVE_GT_FLAG = 1
				save_gts_filename = (sys.argv[i+1])
			elif opt_arg == '--det-method':
				DetectionPlacementFlag = int(sys.argv[i+1])
			elif opt_arg == '--match-thresh':
				MATCH_THRESH = float(sys.argv[i+1])
			elif opt_arg == '--eval-method':
				EVAL_METHOD_FLAG = int(sys.argv[i+1])
			elif opt_arg == '--tol':
				WIN_TOLERANCE = float(sys.argv[i+1])
			else:
				print("\nINVALID INPUT: EXPECTS OPTIONAL INPUTS FROM THE FOLLOWING:")
				print("'--stride [float]'")
				print("'--resamp [int]'")
				print("'--save-preds [filename]'")
				print("'--save-dets [filename]'")
				print("'--save-gts [filename]'\n\n")

				raise("invalid input") # Error out 
			# end of switch statement
			i+=2 # Move on to next pair of optional arguments
		# end of for i
	# end of if optional arguments were passed
	
	#################################
	###  OTHER INPUT DEFINITIONS  ###
	#################################
	
	WINDOWS_PER_BATCH = 128 # Batch-size for the model
	
	if EVAL_METHOD_FLAG == 2 or EVAL_METHOD_FLAG == 3:
		GT_WINDOW_FLAG = 1 #mark that we want Windows from GT if using a window metric
	else:
		GT_WINDOW_FLAG = 0 # mark that we want time points for GT metrics
	# end of Window vs Point GT Flag
	
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
		print("\t1 = Dnd OneHand OREBA")
		print("\t2 = Dnd TwoHand OREBA")
		print("\t3 = Dnd Clemson Data")
		print("\t4 = Dom OneHand OREBA")
		print("\t5 = Dom Clemson Data")
		print("Defaulting to use of Dnd Clemson...")
		DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_Clemson.pkl"
	#end of Database switch statement

	if DEBUG_PRINTS == 1:
		print('CUT_sec Input is ',CUT_sec)
		print('STRIDE_sec input is ',STRIDE_sec)
		print('Eval Data coming from file ',DATABASE_FILEPATH)
		print('RR_Flag Input is ',RR_FLAG)
		print('Fold_Index Input is ',FOLD_INDEX)
		print('Folds_Total Input is ',FOLDS_TOTAL)
		print('Resample Rate Input is ',RESAMPLE_FLAG_FREQ)
		print('Eval Method is ',EVAL_METHOD_NAMES[EVAL_METHOD_FLAG])
		print('WIN_TOLERANCE (if needed) is ',WIN_TOLERANCE)
		print("python setup complete")
	# end of DEBUG_PRINTS
	
	
	############################################
	###  PARSE EVALUATION DATA & LOAD MODEL  ###
	############################################
	
	if TIMING_PRINT == 1:
		print("Parsing Eval Data...")
		start=time.time()
	# end of TIMING_PRINT
	
	if DATABASE_FLAG == 1 or DATABASE_FLAG == 2 or DATABASE_FLAG == 4: # if OREBA
		if RESAMPLE_FLAG_FREQ == 0:
			DataFreq = 64 # Frequency of Data Collection in [Hz]
		else:
			DataFreq = RESAMPLE_FLAG_FREQ # New [Hz] to which to convert signal
		compiled_eval_data = GenerateEvalData_OREBA(
				int(round(CUT_sec*DataFreq)), int(round(STRIDE_sec*DataFreq)),
				DATABASE_FILEPATH,
				FOLD_INDEX,	FOLDS_TOTAL, FOLD_SPLIT=RR_FLAG,
				RESAMPLE_FLAG=RESAMPLE_FLAG_FREQ,
				GT_WINDOW_FLAG = GT_WINDOW_FLAG
				)
		# NOTE: Pass GT_WINDOW_FLAG into OREBA because the original GT is given in Windows
	elif DATABASE_FLAG == 3 or DATABASE_FLAG == 5: # if Clemson
		if RESAMPLE_FLAG_FREQ == 0:
			DataFreq = 15 # Frequency of Data Collection in [Hz]
		else:
			DataFreq = RESAMPLE_FLAG_FREQ # New [Hz] to which to convert signal
		compiled_eval_data = GenerateEvalData_ClemCafe(
				int(round(CUT_sec*DataFreq)), int(round(STRIDE_sec*DataFreq)),
				DATABASE_FILEPATH,
				FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=RR_FLAG,
				RESAMPLE_FLAG=RESAMPLE_FLAG_FREQ # Frequency of Data Collection in [Hz]
				#, RESAMPLE_FLAG=0, SMOOTHING=0 # optional flags not currently used for Paper Experiment
				)
		# NOTE: Pass GT_WINDOW_FLAG into OREBA because the original GT is given in Windows
		if GT_WINDOW_FLAG == 1:
			for meal_num in range(len(compiled_eval_data)):
				compiled_eval_data[meal_num][3] = TimePoint2Window(np.array(compiled_eval_data[meal_num][3])) # default is 2.5 seconds per bite, centered
		# end of if Converting GT to Windows
	#end of Database switch statement
	# Set values to minimum threshold now that data frequency has been defined 
	if CUT_sec == 0:
		CUT_sec = 1/DataFreq
	if STRIDE_sec == 0 :
		STRIDE_sec = 1/DataFreq
	
	if TIMING_PRINT == 1:
		end=time.time()
		print("...Eval data parsed in ",end-start," seconds")
		
		print("Loading Model and Predicting Data...")
		start=time.time()
	# end of TIMING_PRINT
	
	# load keras model
	model = tf.keras.models.load_model(MODEL_NAME)

	
	########################
	###  RUN EVALUATION  ###
	########################
	
	
	All_TP=[] # List (one entry per file) of bites that trigger within ground truth
	All_FP=[] # List of Bites that trigger but have already been mapped to a TP
	All_FN=[] # List of GT windows without a bite detected
	
	for currEvalIndex in range(len(compiled_eval_data)):
		
		meal_num,features_input,timestep_ref,gt_bite_times = compiled_eval_data[currEvalIndex]
		
		##############################################
		##### Bite Count portion of Eval (bc.py) #####
		##############################################
		
		predictions = model.predict(features_input)
		#predictions = np.ones([len(features_input),1]) # quick fake predicitions for debugging
		if SCALE_FACTOR_FOR_BITE_COUNTS > 0:
			predictions = predictions / SCALE_FACTOR_FOR_BITE_COUNTS
		# end if SCALE_FACTOR_FOR_BITE_COUNTS
		
		if SAVE_PREDICITONS_FLAG == 1:
			csvfilename = save_preds_filename + "_Pred_MN" + "{:03}".format(meal_num) + ".csv"
			with open(csvfilename,'w') as fp:
				for i in range(len(predictions)):
					fp.write("{:.4f},{:.4f}\n".format(predictions[i,0],timestep_ref[i]))
				# end of for loop
			#end of with fp
		#end if SAVE_PREDICTIONS_FLAG
		
		
		if TIMING_PRINT == 1:
			end=time.time()
			print("...Predictions made in ",end-start," seconds")
			print("Calculating bite locations from predictions...")
			start=time.time()
		# end of TIMING_PRINT
		
		# Convert Prediction Counts into detections of events
		if DetectionPlacementFlag == 1:
			Detections = Cnt2Detect_MidpointTriggerMidpointPlacement(predictions, timestep_ref, STRIDE_sec, CUT_sec)
		elif DetectionPlacementFlag == 2:
			Detections = Cnt2Detect_TetrisBlockFilling(predictions, timestep_ref, STRIDE_sec, CUT_sec)
		elif DetectionPlacementFlag == 3:
			Detections = Cnt2Detect_TemplateMatching(predictions, timestep_ref, STRIDE_sec, CUT_sec, MATCH_THRESH=MATCH_THRESH)
		elif DetectionPlacementFlag == 4:
			Detections = Cnt2Detect_TetrisBlockFilling_Midpt(predictions, timestep_ref, STRIDE_sec, CUT_sec)
		# end of if Cnt2Detect
		
		if TIMING_PRINT == 1:
			end=time.time()
			print("...Bites Placed in ",end-start," seconds")
		# end of TIMING_PRINT
		
		##############################################
		#### Evaluation Portion of Eval (eval.c) #####
		##############################################
		
		# Convert to np-array to use Eval function 
		Detections = np.array(Detections)
		gt_bite_times = np.array(gt_bite_times)
		
		if SAVE_DETECTIONS_FLAG == 1:
			csvfilename = save_dets_filename + "_Det_MN" + "{:03}".format(meal_num) + ".csv"
			with open(csvfilename,'w') as fp:
				for i in range(len(Detections)):
					fp.write("{:.3f}\n".format(Detections[i]))
				# end of for loop
			#end of with fp
		#end if SAVE_PREDICTIONS_FLAG
		if SAVE_GT_FLAG == 1:
			csvfilename = save_gts_filename + "_GT_MN" + "{:03}".format(meal_num) + ".csv"
			with open(csvfilename,'w') as fp:
				for i in range(len(gt_bite_times)):
					if GT_WINDOW_FLAG == 1:
						fp.write("{:.3f},{:.3f}\n".format(gt_bite_times[i,0],gt_bite_times[i,1]))
					else:
						fp.write("{:.3}\n".format(gt_bite_times[i]))
				# end of for loop
			#end of with fp
		#end if SAVE_PREDICTIONS_FLAG
		
		if len(Detections) == 0:
			print("NO DETECTIONS! {}=FN, mealNum={}".format(len(gt_bite_times), meal_num))
			TP = 0
			FP = 0
			FN = len(gt_bite_times)
		elif EVAL_METHOD_FLAG == 1:
			[TP, FP, FN, Key] = DongEval_PtVsPt(Detections, gt_bite_times)
		elif EVAL_METHOD_FLAG == 2:
			# GT Should have already be converted to Windows
			[TP, FP, FN, FP1, FP2, Key] = KyritEval_PtVsWindow(Detections, gt_bite_times,WIN_TOLERANCE = WIN_TOLERANCE)
		elif EVAL_METHOD_FLAG == 3:
			# Convert Predictions to windows
			Detections_wind = TimePoint2Window(Detections)
			# GT Should have already be converted to Windows
			[TP, FP, FN, TN, Key] = Duration_WindowVsWindow(Detections_wind, gt_bite_times)
		else:
			print("ERROR: UNKNOWN EVAL METHOD REQUESTED. Please pass in 1 for Dong_PtVSPt, 2 for Kyritsis_PtVsWindow, and 3 for Duration_WindVsWind.")
			raise Exception("ERROR: UNKNOWN EVAL METHOD REQUESTED. Please pass in 1 for Dong_PtVSPt, 2 for Kyritsis_PtVsWindow, and 3 for Duration_WindVsWind.")
		#end of EVAL_METHOD_FLAG Switch



		All_TP.append(TP)
		All_FP.append(FP)
		All_FN.append(FN)
		
		if DEBUG_PRINTS == 1:
			if currEvalIndex % 5 == 0: #Status Print
				print("Through {} of {} meals...".format(currEvalIndex+1, len(compiled_eval_data)))
				
			# # Add Results to the lists for each file
			print("MEAL = {}, Metrics = {},{},{}".format(meal_num,TP,FP,FN)) 
			# A DEBUG print for how modular changes results in different numbers
			# # If searching for a specific meal with known good or bad performance
			# if (TP ==  54 and FP == 3 and FN == 13):
			# 	print("meal_num = {}".format(meal_num))
		# end of DEBUG_PRINTS
	
	# end of for currEvalEntry() Loop
	
	
	##############################################
	#### Final Compilation of Results (summ.c) ###
	##############################################
	
	TP = sum(All_TP) # overwrite the short names with the total results
	FP = sum(All_FP)
	FN = sum(All_FN)
	
	# Compute Precision (TPR) Recall (PPV), and F1 Score 
	if (TP+FN)!=0:
		TPR=(float(TP)/(float(TP+FN)))*100
	else:
		TPR=-1 # mark as an Error
		
	if (TP+FP)!=0:
		PPV=(float(TP)/(float(TP+FP)))*100
	else:
		PPV=-1 #mark as an Error
	
	if (TPR+PPV)!=0:
		F1_score=2*TPR*PPV/(TPR+PPV)
	else:
		F1_score=-1
	
	
	
	
	if False: # Verbose Print
		print('TP={0:0d} FP={1:0d} U={2:0d}'.format(TP,FP1+FP2,FN))
		print('\tFP1={0:0d} FP2={1:0d}'.format(FP1,FP2))
		print('TPR={0:.3f}'.format(TPR))
		print('PPV={0:.3f}'.format(PPV))
		print('F1_Score={0:.3f}'.format(F1_score))
		print("Eval Method = {:s}".format(EVAL_METHOD_NAMES[EVAL_METHOD_FLAG])) 
		if PLACEMENT_METHOD_FLAG == 1:
			print('SW Placement=Offset, Offset = {} sec'.format(PLACEMENT_METHOD_1_OFFSET))
		elif PLACEMENT_METHOD_FLAG == 2:
			print('SW Placement=Midpoint')
		#end of PLACEMENT_METHOD_FLAG switch
			
	else: # Single Line Print for Mass Testing
		if DetectionPlacementFlag == 1:
			print('{0:.3f} {1:.3f} {2:.3f} {3:0d} {4:0d} {5:0d} {6} {7:0d} Eval={8:s} WIN_TOL={9:.1f} Place=Midpt'.format(
				F1_score,  TPR, PPV, TP, FP, FN,
				MODEL_NAME, FOLD_INDEX, EVAL_METHOD_NAMES[EVAL_METHOD_FLAG], WIN_TOLERANCE))

		elif DetectionPlacementFlag == 2:
			print('{0:.3f} {1:.3f} {2:.3f} {3:0d} {4:0d} {5:0d} {6} {7:0d} Eval={8:s} WIN_TOL={9:.1f} Place=Tetris'.format(
				F1_score,  TPR, PPV, TP, FP, FN,
				MODEL_NAME, FOLD_INDEX, EVAL_METHOD_NAMES[EVAL_METHOD_FLAG],WIN_TOLERANCE))
		elif DetectionPlacementFlag == 3:
			print('{0:.3f} {1:.3f} {2:.3f} {3:0d} {4:0d} {5:0d} {6} {7:0d} Eval={8:s} WIN_TOL={9:.1f} Place=Template'.format(
				F1_score,  TPR, PPV, TP, FP, FN,
				MODEL_NAME, FOLD_INDEX, EVAL_METHOD_NAMES[EVAL_METHOD_FLAG],WIN_TOLERANCE))
		elif DetectionPlacementFlag == 4:
			print('{0:.3f} {1:.3f} {2:.3f} {3:0d} {4:0d} {5:0d} {6} {7:0d} Eval={8:s} WIN_TOL={9:.1f} Place=TetrisV2'.format(
				F1_score,  TPR, PPV, TP, FP, FN,
				MODEL_NAME, FOLD_INDEX, EVAL_METHOD_NAMES[EVAL_METHOD_FLAG],WIN_TOLERANCE))
		else:
			print("ERROR: NO PLACEMENT METHOD PROPERLY SELECTED; SET EVAL_METHOD_FLAG TO ACCEPTABLE VALUE.")
	
	
# end of main() 