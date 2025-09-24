'''
File: ReimpCTC_Eval.py
Author: James Jolly
Purpose: 
	Takes the logits and predictions from a CTC Reimplementation and then 
	calculates performance metrics
'''




####################
##     Imports    ##
####################

import numpy as np
import time
import sys
import csv
import os

from io import UnsupportedOperation
import pickle as pkl


# Add shared location for auxillary functions
sys.path.insert(1, './../AuxillaryFunctions')

# import personal Functions
from EvaluationMethodFuncs import KyritEval_PtVsWindow
from EvaluationMethodFuncs import DongEval_PtVsPt
from EvaluationMethodFuncs import TimePoint2Window
from EvaluationMethodFuncs import Window2TimePoint
from EvaluationMethodFuncs import PrintStats_SingleLine
from GetResultFilenames import GetResultFilenames


#####################
##   Definitions   ##
#####################


def ReadPredictionsFileForDetections(My_File, DatasetFlag):
	with open(My_File, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		cnt=0
		Detections=[]
		for row in spamreader:
			cnt+=1
			if (row[4]!='1'): # Fifth Column is either 1 for null action or 2 for intake event
				if DatasetFlag == 1 or DatasetFlag== 2 or DatasetFlag == 4: # OREBA Data at 8 Hz
					Detections.append(cnt/8)
				elif DatasetFlag == 3 or DatasetFlag == 5: # Clemson Data at 7.5 hz
					Detections.append(cnt/7.5)
				else:
					print("Invalid DatasetFlag Value when Reading Prediction File... Value given is {} but needs to be in range [1,5].".format(DatasetFlag))
					return []
				# end of switch Dataset
			#end if row is intake
		#end for row
	#end with open(.csv)

	return Detections
# end of ReadPredictionsFileForDetections()




#####################
##   Main Func()   ##
#####################

if __name__ == '__main__':

	##### User Provided Input #####
	Database_Flag = int(sys.argv[1])
		# 1 = Dnd One Hand OREBA
		# 2 = Dnd Two Hand OREBA
		# 3 = Dnd Clemson
		# 4 = Dom One Hand OREBA
		# 5 = Dom Clemson

	# Float value used to determine how many sec the detection can be away from the window
	EvalSelect = int(sys.argv[2]) # 1 = Dong Eval, 2 = Kyritsis Window Tol
	WIN_TOLERANCE = float(sys.argv[3]) # given in [sec]

	PredDirPath = sys.argv[4]

	
	Database_Names = ["None Selected", "Dnd One-Hand OREBA", "Dnd Two-Handed OREBA", "Dnd Clemson", "Dom OREBA One Hand (Dom GT Only)", "Dom Clemson(Dom GT Only)"]
	Database_Acronym = ["", "DndOHO", "DndTHO", "DndClemson", "DomOHO", "DomClemson"]

	# Outputs all detections (Boolean)
	SAVE_DETECTIONS_TO_FILES = True
	if SAVE_DETECTIONS_TO_FILES == True:
		PredFolderName=PredDirPath.split('/')[-2] # grab folder name
		DetectionFolderName="./Detections/{}/{}/".format(
			Database_Acronym[Database_Flag], PredFolderName)
		try:
			print(DetectionFolderName)
			os.mkdir(DetectionFolderName)
			print("Sending Detections to new folder {}...".format(DetectionFolderName))
		except:
			print("sending Detections to existing folder {}...".format(DetectionFolderName))
		# end of try
	# end of if
	# Flag used to print individual file results per course/meal
	PRINTS_INDIVIDUAL_RESULTS = False

	
	# Define Other Variables
	ResultsFilesAllFolds = GetResultFilenames(Database_Flag)

	if Database_Flag == 1: # Dnd OHO
		DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_OneHandOreba.pkl"
	elif Database_Flag == 2: # Dnd THO
		DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_TwoHandOreba.pkl"
	elif Database_Flag == 3: # Dnd Clem
		DATABASE_FILEPATH = "./../Pickle_Databases/Dnd_Clemson.pkl"
	elif Database_Flag == 4: # Dom OHO
		DATABASE_FILEPATH = "./../Pickle_Databases/Dom_OneHandOreba.pkl"
	elif Database_Flag == 5: # Dom Clem
		DATABASE_FILEPATH = "./../Pickle_Databases/Dom_Clemson.pkl"
	else:
		print("Invalid Database_Flag.")
	# end of switch Database_Flag


	with open(DATABASE_FILEPATH,'rb') as fh:
		dataset = pkl.load(fh)
	#OREBA_Cucumber={'UniqueID','proc_data','handedness','bites_gt'}

	UniqueIDs = dataset['UniqueID'] 	



	# Global Results from all folds (each element is a list from each fold)
	All_All_TP = []
	All_All_FP = []
	All_All_FN = []


	for currFold in range(5):

		# Fold number to evaluate. Valid range is [0,4]
		FOLD_SELECT = currFold

		# Directory containing all results for the specified fold
		# PredDirPath = './FiveFold_CTCResults_v1/Clem_Intake/fold{}/'.format(FOLD_SELECT)
		PredDirPathFold = PredDirPath + 'Fold{}/'.format(FOLD_SELECT)

		# Select Current Fold Results Filenames
		ResultsFiles = ResultsFilesAllFolds[FOLD_SELECT]

		# Create Sanity Check of All Participant and Meal IDs from filenames 
		#     to check against pickle Unique IDs later
		mealID_Check = []
		for filename in ResultsFiles:
			filenameComponents = filename.split("_")
			if Database_Flag == 1 or Database_Flag == 2 or Database_Flag == 4:
				mealID_Check.append(filenameComponents[-5] + '_' + filenameComponents[-4])
			else:
				mealID_Check.append(filenameComponents[-5] + filenameComponents[-4])
			# end if
		# end of for filename




		# Define blank sets for results
		All_TP=[]
		All_FP=[]
		All_FN=[]

		for idx in range(0, len(ResultsFiles)):
			currFilename = PredDirPathFold + ResultsFiles[idx]
			currFileID = mealID_Check[idx]

			# NEED TO SEARCH FOR PICKLE FILE SINCE PICKLE CONTAINS EXTRA MEALS NOT IN CTC FOLD
			try:
				pickle_idx = UniqueIDs.index(currFileID) # get current file index in pickle dataset
			except ValueError:
				print("UNABLE TO FIND PICKLE MEAL CORRESPONDING TO MEAL {}.".format(mealID_Check[idx]))
				continue
			#end of try statement
			currPickleID = UniqueIDs[pickle_idx]

			if currPickleID != currFileID:
				print("ERROR: Mismatched FileID ({}) and MealID ({}).".format(currFileID, currPickleID))
			# else:
			# 	print("Matching FileID for {}...".format(currFileID))
			# end of Pickle ID Sanity check

			currDets = np.array(ReadPredictionsFileForDetections(currFilename, Database_Flag))

			if Database_Flag == 1 or Database_Flag == 2 or Database_Flag == 4: # if OREBA
				if EvalSelect==1: # Using Dong Eval
					currGT = Window2TimePoint(dataset['bites_gt'][pickle_idx])
				elif EvalSelect==2: # Using Kyritsis Eval with tolerance
					currGT = dataset['bites_gt'][pickle_idx]
				# end EvalSelect switch
			else: # If Clemson Dataset with Pt GT
				if EvalSelect==1: # Using Dong Eval
					currGT = dataset['bites_gt'][pickle_idx]/15
				elif EvalSelect==2: # Using Kyritsis Eval with tolerance
					currGT = TimePoint2Window(dataset['bites_gt'][pickle_idx]/15)
				# end EvalSelect switch
			# end of Database_Flag switch



			if len(currDets) == 0:
				# print("NO DETECTIONS! {}=FN, mealNum={}".format(len(currGT), meal_num))
				TP = 0
				FP = 0
				FN = len(currGT)
			elif EvalSelect==1: # Using Dong Eval
				[TP, FP, FN, _] = DongEval_PtVsPt(currDets, currGT)
			elif EvalSelect==2: # Using Kyritsis Eval with tolerance
				[TP, FP, FN, FP1, FP2, _] = KyritEval_PtVsWindow(currDets, currGT, WIN_TOLERANCE=WIN_TOLERANCE)
				# [TP, FP, FN, FP1, FP2, Key] = KyritEval_PtVsWindow(currDets, currGT, WIN_TOLERANCE = 0.0, BEFORE_TOL = 0.0, AFTER_TOL = 0.0):
			# end of EvalSelect switch 

			All_TP.append(TP)
			All_FP.append(FP)
			All_FN.append(FN)
			
			if SAVE_DETECTIONS_TO_FILES:
				currID = currPickleID
				
				
				filename = DetectionFolderName + "Dets_{}_{}.txt".format(
					Database_Acronym[Database_Flag], currID)
				with open(filename,"w") as f:
					Header = "Dataset = {}; Detections given in seconds from start of data\n".format(
						Database_Names[Database_Flag]
					)
					f.write(Header)

					TextDetections=[]
					for currDetection in currDets:
						TextDetections.append("{:0.2f}".format(currDetection)) # save all detections to 2 decimals
					# end for
					FullDetectionText = '\n'.join(TextDetections)+'\n'
					f.write(FullDetectionText)
				# end of with open()
			# end of if SAVE_DETECTIONS_TO_FILES
			

			if PRINTS_INDIVIDUAL_RESULTS == True:
				print("\t",end='')
				PrintStats_SingleLine(TP, FP, FN)
			# end Print Individual Results
		# end of for idx

		All_All_TP.append(All_TP)
		All_All_FP.append(All_FP)
		All_All_FN.append(All_FN)


	# end of for currFold



	#### Calculate total performance on data
	Final_TP = 0
	Final_FP = 0
	Final_FN = 0

	for i in range(5):
		Final_TP += sum(All_All_TP[i]) 
		Final_FP += sum(All_All_FP[i]) 
		Final_FN += sum(All_All_FN[i])
	# end of for fold loop 

	print("Results for dataset {}, EvalSelect = {}, Tol = {}".format(
		Database_Acronym[Database_Flag],
		EvalSelect, WIN_TOLERANCE))
	PrintStats_SingleLine(Final_TP, Final_FP, Final_FN, VerbosePrintOpt = 1)




# end of main()
