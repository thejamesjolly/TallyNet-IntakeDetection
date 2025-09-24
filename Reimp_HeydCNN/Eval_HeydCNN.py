import sys
import os
import math
import numpy as np # v1.19.5
import tensorflow as tf # version 2.2.0
from tensorflow import keras


from io import UnsupportedOperation
import pickle as pkl


# Add shared location for auxillary functions
sys.path.insert(1, './../AuxillaryFunctions')


#import matplotlib.pyplot as plt

from GenerateClassDataFuncs import GenerateEvalData_OREBA
from GenerateClassDataFuncs import GenerateEvalData_ClemCafe


from EvaluationMethodFuncs import DongEval_PtVsPt
from EvaluationMethodFuncs import KyritEval_PtVsWindow
from EvaluationMethodFuncs import TimePoint2Window
from EvaluationMethodFuncs import Window2TimePoint
from EvaluationMethodFuncs import Duration_WindowVsWindow
from EvaluationMethodFuncs import PrintStats_SingleLine













if __name__ == '__main__':
	DEBUG_PRINTS = 0

	MODEL_FILE = sys.argv[1]
	DATABASE_FLAG = int(sys.argv[2]) # Must be 1 (Dom Hand Only) or 2 (Both Hands)
	FOLDS_TOTAL = 5 # total number of folds to split the data into
	FOLD_INDEX = int(sys.argv[3]) % FOLDS_TOTAL # Which index to leave out for validation

	# # # # # Use model Predictions to place bite in Window # # # # #
	BiteDetectThresh=float(sys.argv[4]) # Value used in Paper to trigger a detection
	bite_length_sec = 2 # duration in sec to mark a triggered detection as 
						# a bite and ignore any value in that window
	delayed_offset_sec = 0.0
	# add a small offset to center the predictions in window

	DATA_FREQ=64 # Heyd Model Always at 64 Hz

	RR_FLAG = 1 # 1 if using striped fold division, 0 if using block fold division
	WIN_TOLERANCE = 8 # seconds of time before and after bite to fall within gesture window


	CUT_sec = 2 # window is always 128 (2 x 64 hz)
	STRIDE_sec = float(6.0/64.0) 
	STRIDE = STRIDE_sec * DATA_FREQ


	# default to 0.0 for strict method, otherwise provide [sec] preds can be away from window


	# NUMBER OF HANDS
	if DATABASE_FLAG == 1:
		DATABASE_FILEPATH = './../Pickle_Databases/Dnd_OneHandOreba.pkl'
		RESAMPLE_FLAG_FREQ = 0 # zero if keeping original data frequency
	elif DATABASE_FLAG == 2:
		DATABASE_FILEPATH = './../Pickle_Databases/Dnd_TwoHandOreba.pkl'
		RESAMPLE_FLAG_FREQ = 0 # zero if keeping original data frequency
	elif DATABASE_FLAG == 3:
		DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_Clemson.pkl"
		RESAMPLE_FLAG_FREQ = 64 # new freq since ClemCafe = 15Hz
	elif DATABASE_FLAG == 4:
		DATABASE_FILEPATH = './../Pickle_Databases/Dom_OneHandOreba.pkl'
		RESAMPLE_FLAG_FREQ=0 # zero if keeping original data frequency
	elif DATABASE_FLAG == 5:
		DATABASE_FILEPATH = './../Pickle_Databases/Dom_Clemson.pkl'
		RESAMPLE_FLAG_FREQ = 64 # new freq since ClemCafe = 15Hz
	### Removed Dataset using ~12 additional meals from Clemson Dataset which had Point GT labels but not gesture labels
	# elif DATABASE_FLAG == 6:
	# 	DATABASE_FILEPATH = './../Pickle_Databases/CleanClemCafe.pkl'
	# 	RESAMPLE_FLAG_FREQ = 64 # new freq since ClemCafe = 15Hz
	else:
		print("!!! WARNING: INVALID DATABASE SELECTION FLAG VALUE OF {} !!!".format(DATABASE_FLAG))
		print("Expects value of:")
		print("\t1 = Dnd OneHand OREBA")
		print("\t2 = Dnd TwoHand OREBA")
		print("\t3 = Dnd ClemCafe Data")
		print("\t4 = Dom Clemson")
		print("\t5 = Dom OneHand OREBA")
		# print("\t6 = PointGT Clemson (Unused in TallyNet Paper)")

		exit(0)
	#end of Database switch statement



	Database_Names = ["None Selected", "Dnd One-Hand OREBA", "Dnd Two-Handed OREBA", "Dnd Clemson", "Dom OREBA One Hand (Dom GT Only)", "Dom Clemson(Dom GT Only)"]
	Database_Acronym = ["", "DndOHO", "DndTHO", "DndClemson", "DomOHO", "DomClemson"]

	# Outputs all detections (Boolean)
	SAVE_DETECTIONS_TO_FILES = True
	if SAVE_DETECTIONS_TO_FILES == True:
		PredFolderName=MODEL_FILE.split('/')[-3] # grab folder name
		DetectionFolderName="/scratch/jpjolly/HeydDets/Detections/{}/{}/".format(
			Database_Acronym[DATABASE_FLAG], PredFolderName)
		try:
			print(DetectionFolderName)
			os.mkdir(DetectionFolderName)
			print("Sending Detections to new folder {}...".format(DetectionFolderName))
		except:
			print("sending Detections to existing folder {}...".format(DetectionFolderName))
		# end of try
		
		with open(DATABASE_FILEPATH,'rb') as fh:
			dataset = pkl.load(fh)
		UniqueIDs = dataset['UniqueID'] 
	# end of if

	# OLD VERSION OF BITE DETECTOR
	EVAL_METHODS_TO_USE = [1,2]
	# EVAL_METHOD_FLAG = 1
		# 1 = Dong Eval (Pt vs Pt)
		# 2 = Kyritsis (Pt vs Window)
		# 3 = Duration (Window vs Window)

	GT_WINDOW_FLAG = 1
	# if EVAL_METHOD_FLAG == 2 or EVAL_METHOD_FLAG == 3:
	# 	GT_WINDOW_FLAG = 1 #mark that we want Windows from GT if using a window metric
	# else:
	# 	GT_WINDOW_FLAG = 0 # mark that we want time points for GT metrics
	# # end of Window vs Point GT Flag


	# # # # # # KNOBS TO TURN FOR EVAL METHOD # # # # # #
	DETECTION_METHOD_FLAG = 1 
		# Used to determine how Local Maximums are determined
		# 1 = Immediate Local Maximum; just considers points around it;
		#       Only triggers once every bite_length_sec
		# 2 = Searches each non-floored segment to find the 
		#       maximum and places the detection there; see segment for additional knobs





	EVAL_METHOD_NAMES = ["Names for Eval methods (valid flags start at 1)",
				 "Dong_PtVsPt",
				 "Kyritsis_PtVsWind",
				 "Duration_WindVsWind"]

	if DEBUG_PRINTS == 1:


		print('CUT_sec Input is ',CUT_sec)
		print('STRIDE_sec input is ',STRIDE_sec)
		print('Training Data coming from file ',DATABASE_FILEPATH)
		print('RR_Flag Input is ',RR_FLAG)
		print('Fold_Index Input is ',FOLD_INDEX)
		print('Folds_Total Input is ',FOLDS_TOTAL)
		print('Resample Rate Input is ',RESAMPLE_FLAG_FREQ)
		print("Eval Methods to use:")
		for MethodIdx in EVAL_METHODS_TO_USE:
			print('\t{}',EVAL_METHOD_NAMES[MethodIdx])
		print('WIN_TOLERANCE (if needed) is ',WIN_TOLERANCE)
		print('Offset_sec (if needed) is ',delayed_offset_sec)
		print("python setup complete")
	# end of DEBUG_PRINTS


	# # # # # Read in file eval data # # # # #
	if DATABASE_FLAG == 1 or DATABASE_FLAG == 2 or DATABASE_FLAG == 4: # OREBA datasets
		compiled_eval_data = GenerateEvalData_OREBA(
				int(round(CUT_sec*DATA_FREQ)), int(round(STRIDE_sec*DATA_FREQ)),
				DATABASE_FILEPATH,
				FOLD_INDEX,	FOLDS_TOTAL, FOLD_SPLIT=RR_FLAG,
				RESAMPLE_FLAG=RESAMPLE_FLAG_FREQ,
				GT_WINDOW_FLAG = GT_WINDOW_FLAG
				)
		pt_gt_data=[]
		for meal_num in range(len(compiled_eval_data)):
			pt_gt_data.append(Window2TimePoint(np.array(compiled_eval_data[meal_num][3])))
		# NOTE: Pass GT_WINDOW_FLAG into OREBA because the original GT is given in Windows
	elif DATABASE_FLAG == 3 or DATABASE_FLAG == 5: # Clemson Dataset
		compiled_eval_data = GenerateEvalData_ClemCafe(
				int(round(CUT_sec*DATA_FREQ)), int(round(STRIDE_sec*DATA_FREQ)),
				DATABASE_FILEPATH,
				FOLD_INDEX, FOLDS_TOTAL, FOLD_SPLIT=RR_FLAG,
				RESAMPLE_FLAG=RESAMPLE_FLAG_FREQ # Frequency of Data Collection in [Hz]
				#, RESAMPLE_FLAG=0, SMOOTHING=0 # optional flags not currently used for Paper Experiment
				)
		if GT_WINDOW_FLAG == 1:
			# Save Point GT as separate variable
			pt_gt_data=[]
			for meal_num in range(len(compiled_eval_data)):
				pt_gt_data.append(compiled_eval_data[meal_num][3])
			# Convert remaining Data to Windows
			for meal_num in range(len(compiled_eval_data)):
				compiled_eval_data[meal_num][3] = TimePoint2Window(np.array(compiled_eval_data[meal_num][3])) # default is 2.5 seconds per bite, centered
		# end of if Converting GT to Windows
	else:
		print("Invalid Database flag value of {}".format(DATABASE_FLAG))
		raise ValueError("Invalid Database flag")
	#end of Database switch statement

	# # # # # Load Network Model # # # # #
	model = tf.keras.models.load_model(MODEL_FILE)



	# # # # # Construct Bite Predictions for Bite Indices # # # # #
	if DEBUG_PRINTS == 1:
		print("Testing...")



	All_Detections = []
	All_gt_bite_times = []

	for curr_meal in range(len(compiled_eval_data)):

		# Grab current meal data
		[meal_id,features_input,timesteps,currGT_times]=compiled_eval_data[curr_meal] 
		# Get prediction probabilities
		predictions = model.predict(features_input)
		if DEBUG_PRINTS==1:
			predictions.tofile('Predictions_Raw'+str(curr_meal)+'_subj'+str(FOLD_INDEX)+'.csv', sep = ',')
			np.array(timesteps).tofile('timesteps'+str(curr_meal)+'_subj'+str(FOLD_INDEX)+'.csv', sep = ',')
		# End if DEBUG_PRINT
		for i in range(len(predictions)):
			if predictions[i]<BiteDetectThresh:
				predictions[i]=0
		# predictions[predictions<BiteDetectThresh]=0 # floor any predictions below threshold


		if DETECTION_METHOD_FLAG == 1: # First Method: Floor predictions, then find any local maximum (based on its two neighbors), 
			#then detect the largest LM that ensures all LM are either detections or covered up in 2 sec window 

			# Filter out so only local maxima remain
			predictions_LM=[]
			#take care of front edge cases
			if predictions[0]>predictions[1]:
				predictions_LM.append(1)
			else:
				predictions_LM.append(0)

			#iterate over the rest of the data
			for i in range(1,len(predictions)-1):
				#Bias data towards the left most point in the event of equal values
				if predictions[i]>predictions[i-1] and predictions[i]>=predictions[i+1]:
					predictions_LM.append(1)
				else:
					predictions_LM.append(0)
			#take care of back edge cases
			if predictions[-1]!=0 and predictions[-1]>=predictions[-2]:
				predictions_LM.append(1)
			else:
				predictions_LM.append(0)

			predictions_LM=np.array(predictions_LM)


			Detections=[] # initialize as empty list, add detections as they are found
			for i in range(len(predictions_LM)):
				# if it is a local maximum that needs to be either covered or detected
				if predictions_LM[i]==1:
					# find maximum value within duration ahead to see if it will be covered
					running_max_index=i
					#check the duration ahead for a higher local peak
					for j in range(1,round((bite_length_sec*DATA_FREQ/STRIDE)-0.5)):
						if (i+j)>=len(predictions_LM):
							break
						if predictions_LM[i+j]==1:
							if predictions[running_max_index]<predictions[i+j]:
								# erase previous LM since it is "covered up" by new detection placement
								predictions_LM[running_max_index]=0
								# mark new location of running max 
								running_max_index=i+j
							else:
								predictions_LM[i+j]=0 # Cover up the smaller local maxima
					# Add detection, then cover remaining points up to duration seconds ahead
					Detections.append(timesteps[running_max_index]+delayed_offset_sec)
					for j in range(0,round((bite_length_sec*DATA_FREQ/STRIDE)-0.5)):
						if (running_max_index+j)>=len(predictions_LM):
							break
						if predictions_LM[running_max_index+j]==1:
							predictions_LM[running_max_index+j]=0
				# end of if predictions_LM[i]=1
			#end of for i in range(len(predictions_LM)) loop

		elif DETECTION_METHOD_FLAG == 2: # 2nd Attempt: Grab the maximum in each non-floored segment

			### Knobs ###
			SizeOfGapToMergeSegments=5 #number of predictions needing to be a zero to close off a segment

			### Other Variables ###
			FinishedFlag1=0 # used to mark end of loop across all predictions
			rover1=0 # Main iterator marking where in the prediciton file has been fully inspected
			rover2=0 # when main rover finds a segment, rover 2 scouts ahead to find the 
					# first datapoint after the segment (or end of list)
			Detections=[] # initialize as empty list, add detections as they are found

			while (FinishedFlag1!=1):
				if predictions[rover1]==0: #if no detection, move to next bite
					rover1+=1
					if rover1>=len(predictions)-1: #add the "-1" in order to leave space for rover2 to be inserted
						FinishedFlag1=1
				else: #if the prediction is the start of a bite segment
					FinishedFlag2=0
					rover2=rover1+1 #start scout just ahead of current detected start
					while FinishedFlag2!=1:
						if predictions[rover2]==0:
							FinishedFlag2=1 # Begin with assumption one will end loop if this is the end of segment
							# start at one and add up to SizeOfGapToMergeSegments
							for i in range(1,SizeOfGapToMergeSegments+1):
								# set rover to end if the end of the data is reached
								if rover2+i>=len(predictions):
									rover2=len(predictions)-1
									break # leave i loop and end while since Flag is still false
								if predictions[rover2+i]!=0: #if the gap is short and a non-zero bite is found
									rover2=rover2+i # move scout to the non-zero point
									FinishedFlag2=0 # correct assumption since the gap was short
									break # and continue search in while() loop
							# if loop is exited and FinishedFlag2 has not been cleared, then rover2 is at end
						else: # if predictions[rover2]!=0
							rover2+=1 # move rover along if segment is still occuring
							if rover2>=len(predictions): # check boundary for end of data
								rover2=len(predictions)-1
								FinishedFlag2=1 # or use break statement
					# end of while FinishedFlag2!=1


					# Add Detection as the maximum of the segment
					currMax=predictions[rover1] #start with first point as max
					currMaxIndex=rover1
					for rover3 in range(rover1+1,rover2): # iterate over all remaining data in segment
						if predictions[rover3]>=currMax: # biased to later times if a tie
							currMax=predictions[rover3] 
							currMaxIndex=rover3
					Detections.append(timesteps[currMaxIndex]+delayed_offset_sec)

					# Shift Rover to end of Segment and continue
					rover1=rover2
					if rover1>=len(predictions)-1: #set to minus one from end so that rover2=rover1+1 won't exceed index
						FinishedFlag1=1 #could also just use a break statement

			# end of while FinishedFlag==0
		# end of predictions[] --> Detections[] if statement






		##################################
		####  MATCH DETECTIONS TO GT  ####
		##################################


		if SAVE_DETECTIONS_TO_FILES:
			currID = UniqueIDs[meal_id]
			filename = DetectionFolderName + "Dets_{}_{}.txt".format(
				Database_Acronym[DATABASE_FLAG], currID)
			with open(filename,"w") as f:
				Header = "Dataset = {}; Detections given in seconds from start of data\n".format(
					Database_Names[DATABASE_FLAG]
				)
				f.write(Header)

				TextDetections=[]
				for currDetection in Detections:
					TextDetections.append("{:0.2f}".format(currDetection)) # save all detections to 2 decimals
				# end for
				FullDetectionText = '\n'.join(TextDetections)+'\n'
				f.write(FullDetectionText)
			# end of with open()
		# end of if SAVE_DETECTIONS_TO_FILES
		
		# Convert to np-array to use Eval function 
		Detections = np.array(Detections)
		gt_bite_times = np.array(currGT_times)

		All_Detections.append(Detections)
		All_gt_bite_times.append(gt_bite_times)
	# end of for currMeal loop


	for EvalMethod in EVAL_METHODS_TO_USE:
		EVAL_METHOD_FLAG = EvalMethod

		# Clear Values for each eval method
		# Lists for the metrics of each meal evaluated
		All_TP=[] # List (one entry per file) of bites that trigger within ground truth
		All_FP=[] # List of Bites that trigger but have already been mapped to a TP
		All_FN=[] # List of GT windows without a bite detected

		# If using Point vs Window Eval, use these two lists
		All_FP1=[] # List of Bites that trigger inside of another TP
		All_FP2=[] # List of Bites that trigger outside of any GT

		for curr_meal in range(len(compiled_eval_data)):	

			# Convert to np-array to use Eval function 
			Detections = All_Detections[curr_meal]
			gt_bite_times = All_gt_bite_times[curr_meal] 
			curr_pt_gt_data = pt_gt_data[curr_meal]

			if (len(Detections)==0):
				if DEBUG_PRINTS == 1:
					print("Zero Detections for meal index {}".format(curr_meal))
					print("len = {}, shape = ".format(len(gt_bite_times)),end='')
					print(np.shape(gt_bite_times))
				# end of DEBUG_PRINTS
				TP = 0
				FP = 0
				FN = len(gt_bite_times)

			elif EVAL_METHOD_FLAG == 1:
				[TP, FP, FN, Key] = DongEval_PtVsPt(
					Detections, curr_pt_gt_data
				)
			elif EVAL_METHOD_FLAG == 2:
				# GT Should have already be converted to Windows
				[TP, FP, FN, FP1, FP2, Key] = KyritEval_PtVsWindow(
					Detections, gt_bite_times,WIN_TOLERANCE = WIN_TOLERANCE
				)
				All_FP1.append(FP1)
				All_FP2.append(FP2)
			elif EVAL_METHOD_FLAG == 3:
				# Convert Predictions to windows
				Detections_wind = TimePoint2Window(Detections)
				# GT Should have already be converted to Windows
				[TP, FP, FN, TN, Key] = Duration_WindowVsWindow(
					Detections_wind, gt_bite_times
				)
			else:
				print("ERROR: UNKNOWN EVAL METHOD REQUESTED. Please pass in 1 for Dong_PtVSPt, 2 for Kyritsis_PtVsWindow, and 3 for Duration_WindVsWind.")
				raise Exception("ERROR: UNKNOWN EVAL METHOD REQUESTED. Please pass in 1 for Dong_PtVSPt, 2 for Kyritsis_PtVsWindow, and 3 for Duration_WindVsWind.")
			#end of EVAL_METHOD_FLAG Switch



			All_TP.append(TP)
			All_FP.append(FP)
			All_FN.append(FN)
			if DEBUG_PRINTS == 1:
				print("Through meal {} of {}...".format(curr_meal, len(compiled_eval_data)))
			#end of DEBUG_PRINTS
		# end of for curr meal loop

		TP = sum(All_TP) # overwrite the short names with the total results
		FP = sum(All_FP)
		FN = sum(All_FN)

		# print("TP = {}, FP = {}, FN = {}".format(TP,FP,FN))

		# print stats at start of line with eval validation info
		print("BiteThresh = {thresh:0.3f}; Eval={EvalMethod:{maxlen}s}, Dataset={dataset}, Fold={foldidx} of {foldtotal}, WinTol={wintol}; ".format(
				thresh=BiteDetectThresh,
				EvalMethod=EVAL_METHOD_NAMES[EVAL_METHOD_FLAG],
				maxlen=max([len(x) for x in [EVAL_METHOD_NAMES[EvalIdx] for EvalIdx in EVAL_METHODS_TO_USE]]),
				dataset=DATABASE_FILEPATH.split('/')[-1], 
				foldidx=FOLD_INDEX, foldtotal=FOLDS_TOTAL, 
				wintol=WIN_TOLERANCE), end='')
		# Print relevant stats for each eval method
		if EVAL_METHOD_FLAG == 2:
			FP1 = sum(All_FP1)
			FP2 = sum(All_FP2)
			#Verbose flag set to 0
			PrintStats_SingleLine(TP, FP, FN, FP1 = FP1, FP2 = FP2,
					  VerbosePrintOpt = 0
					  )
		else:
			PrintStats_SingleLine(TP, FP, FN)
		# end of PrintStats switch

	# end of for EvalMethod

	

	if DEBUG_PRINTS == 1:
		print("FINISHED EVALUATING")
	#end DEBUG_PRINTS

	#PlotFIC_GT_Preds(currGT_times,gt_matched,predictions,timesteps,Detections,DetectionOutcomes)

# end of main()

