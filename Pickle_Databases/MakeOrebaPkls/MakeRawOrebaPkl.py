'''
File: MakeRawOrebaPkl.py
Author: James Jolly
Purpose: 
	Create a pickle file of the raw IMU Data for the One Hand OREBA dataset 
	using both Dominant and Non-Dominant ground truth gestures as ground truth events.
	This data is not recommended for Machine learning since it is not normalized
	but can be used in the Dong Heuristic Reimplementation as a benchmark method.
	To use this as the Dominant Only data, filter the GT based on hand in evaluation 
	or Grab GT timestamps from DomOreba.pkl
'''




####################
##     Imports    ##
####################
import sys
from io import UnsupportedOperation
import pickle as pkl
import numpy as np
import csv




#####################
##   Definitions   ##
#####################

def TimeStr2Sec(TimeStr):
	#Split up String into real number
	intervals=TimeStr.split(':')
	
	if len(intervals)==3:
		# Convert number strings to actual numbers
		hours=int(intervals[0])
		minutes=int(intervals[1])
		seconds=float(intervals[2])
		# Sum Times to single unit
		FinalTime=float(((hours*60)+minutes)*60)+seconds
	elif len(intervals)==2:
		# Convert number strings to actual numbers
		minutes=int(intervals[0])
		seconds=float(intervals[1])
		# Sum Times to single unit
		FinalTime=float(minutes*60)+seconds
	
	return FinalTime
#end of TimeStr2Sec()







#####################
##   Main Func()   ##
#####################

if __name__ == '__main__':
	
	PROCESSED_DATA_FILEPATHS='OREBA_processed_files.txt'
	ANNOTATIONS_FILEPATHS='OREBA_annotation_files.txt'




	AllDataFiles = []
	with open(PROCESSED_DATA_FILEPATHS) as fpt_files:
			for filename in fpt_files:
				noNewLineName = (filename.splitlines())[0]
				AllDataFiles.append(noNewLineName)

	AllData=[] # initialize empty list
	UniqueID=[]
	DomHand=[]
	AllDataTimes=[]
	AllBiteHandedness=[]


	leftyColumnOffset = 3
	rightyColumnOffset = 9
	DomHandFlagColumn = 15

	cntr=0 # counter used for periodic status print

	for currFileName in AllDataFiles:

		with open(currFileName) as currFile:
			currFile.readline() # Read off Header Row

			meal_data=[]
			data_times=[]
			for line in currFile:
				row=line.split(',')
				if row[DomHandFlagColumn]=="right":
					DomHandOffset = rightyColumnOffset
				elif row[DomHandFlagColumn]=="left":
					DomHandOffset = leftyColumnOffset
				else:
					print("ERROR DETERMINING DOMINANT HAND FLAG")
					print("Given: {}".format(row[DomHandFlagColumn]))
				timestep_data=[]
				for i in range(6):
					timestep_data.append(float(row[DomHandOffset+i]))
				meal_data.append(timestep_data)

				data_times.append(TimeStr2Sec(row[2]))
		# end of currFileName read

		#print("Completed Meal {}".format(currFileName))

		UniqueID.append(row[0])
		DomHand.append(row[15])
		AllData.append(np.array(meal_data))
		AllDataTimes.append(np.array(data_times))


		cntr+=1
		if cntr%5 == 0:
			print("Through meal {} of {}...".format(cntr, len(AllDataFiles)))
		# end of if cntr
	# end of for currFileName loop


	print("Finished With Meals")



	AllAnnotationFiles = []
	with open(ANNOTATIONS_FILEPATHS) as fpt_files:
			for filename in fpt_files:
				noNewLineName = (filename.splitlines())[0]
				AllAnnotationFiles.append(noNewLineName)


	AllBites=[] # initialize empty list

	for currFileName in AllAnnotationFiles:

		with open(currFileName) as currFile:
			currFile.readline() # Read off Header Row

			bite_handedness=[]
			meal_annotation=[]
			for line in currFile:
				row2=line.split(',')
				timestep_annotation=[TimeStr2Sec(row2[0]),TimeStr2Sec(row2[1])]
				bite_hand = row2[6]
				if bite_hand == 'Right':
					bite_handedness.append(0)
				elif bite_hand =='Left':
					bite_handedness.append(1)
				elif bite_hand =='Both':
					bite_handedness.append(2)
				else:
					print("I DON'T KNOW HOW TO HANDLE THIS")
					print(bite_hand)

				meal_annotation.append(timestep_annotation)
		# end of currFileName read

		#print("Completed Meal {}".format(currFileName))
		AllBiteHandedness.append(np.array(bite_handedness))
		AllBites.append(np.array(meal_annotation))
	# end of for currFileName loop


	print("Finished with Annotations")




	README = [["INTRODUCTION",
			   'This pkl dataset contains all raw sensor data from the OREBA Dataset processed for easy use in Python in IMU event tracking.'
			  ],
			  ["DICTIONARY ENTRIES IN DATABASE",
			   'README: Description of Pickle Database',
			   'UniqueID: Contains a participant and course number to uniquely identify each meal file in the original Database.',
			   'raw_domHand_data: Sensor readings as reported in OREBA without any rotation or processing. Given as acc_x, acc_y, acc_z, then gyro_x, gyro_y, gyro_z (pitch roll yaw).',
			   #'data_times: time [sec] from the start of the data, used to index answer.  Can be computed as index/15 since data is at 15Hz.',
			   'handedness: list with an entry for each meal of whether the user was left or right hand dominant, and thus on which hand the data collection device was.',
			   'bites_gt: list of numpy array containing the indices of each intake moment (the specific moment food entered the mouth). Can convert to time since recording started by simply dividing value by 15 (since data is at 15Hz).',
			   'bites_handedness_gt: list of lists, containing whether the corresponding ground truth bite was eaten with the left, right, or both hands.',
			  ]
			 ]


	OREBA_Cucumber={
		'README': README,
		'UniqueID': UniqueID,
		'raw_domHand_data': AllData,
		'data_times': AllDataTimes,
		'handedness': DomHand,
		'bites_gt': AllBites,
		'bites_handedness_gt': AllBiteHandedness
	}

	print("Finished Making Cucumber for Pickling")

	outfile = open("../Raw_OneHandOreba.pkl",'wb')
	pkl.dump(OREBA_Cucumber,outfile)
	outfile.close()

	print("Finished Pickling Raw One Handed OREBA Dataset")




# end of main()
