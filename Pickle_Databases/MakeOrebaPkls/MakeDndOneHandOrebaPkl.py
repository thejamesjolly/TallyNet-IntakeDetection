'''
File: MakeDndClemsonPkl.py
Author: James Jolly
Purpose: 
	Create a pickle file of the One Hand OREBA dataset 
	using both Dominant and Non-Dominant ground truth gestures as ground truth events.
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
#end def TimeStr2Sec()







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

	idx=0
	for currFileName in AllDataFiles:

		with open(currFileName) as currFile:
			currFile.readline() # Read off Header Row

			meal_data=[]
			data_times=[]
			for line in currFile:
				row=line.split(',')
				timestep_data=[]
				for i in range(6):
					timestep_data.append(float(row[3+i]))
				meal_data.append(timestep_data)

				data_times.append(TimeStr2Sec(row[2]))
		# end of currFileName read

		#print("Completed Meal {}".format(currFileName))

		UniqueID.append(row[0])
		DomHand.append(row[15])
		AllData.append(np.array(meal_data))
		AllDataTimes.append(np.array(data_times))

		idx+=1
		if idx % 10 == 0:
			print("Through {} of {} Data Files...".format(idx, len(AllDataFiles)))
		# end of print

	# end of for currFileName loop


	print("Finished With Meals")



	AllAnnotationFiles = []
	with open(ANNOTATIONS_FILEPATHS) as fpt_files:
			for filename in fpt_files:
				noNewLineName = (filename.splitlines())[0]
				AllAnnotationFiles.append(noNewLineName)


	AllBites=[] # initialize empty list

	AllUtensils=[]
	AllDomVsNondom=[]

	idx=0
	for currFileName in AllAnnotationFiles:

		currDomHand = DomHand[idx]
		idx+=1

		with open(currFileName) as currFile:
			currFile.readline() # Read off Header Row

			bite_handedness=[]
			meal_annotation=[]
			utensils=[]
			domVsNondom=[]
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

				if (currDomHand == bite_hand):
					domVsNondom.append(1) # 1 = Dominant
				elif (bite_hand == 'both'): 
					domVsNondom.append(2) # 2 = Both Hands
				else:
					domVsNondom.append(0) # 0 = Non-Dominant
				# end of DomVsNondom split

				bite_utensil=row2[7].lower()
				if bite_utensil == 'cup' or bite_utensil == 'bottle':
					utensils.append(0)
				elif bite_utensil == 'hand':
					utensils.append(1)
				elif bite_utensil == 'fork':
					utensils.append(2)
				elif bite_utensil == 'spoon':
					utensils.append(3)
				elif bite_utensil == 'knife':
					utensils.append(4)
				elif bite_utensil == 'finger':
					utensils.append(5)
				# end of utensil switch

		# end of currFileName read

		#print("Completed Meal {}".format(currFileName))
		AllBiteHandedness.append(np.array(bite_handedness))
		AllBites.append(np.array(meal_annotation))
		AllUtensils.append(np.array(utensils))
		AllDomVsNondom.append(np.array(domVsNondom))

		if idx % 10 == 0:
			print("Through {} of {} Annotations...".format(idx, len(AllAnnotationFiles)))
		# end of print
	# end of for currFileName loop


	print("Finished with Annotations")



	README = [["INTRODUCTION",
			   'This pkl dataset contains all data from the OREBA Dataset (Dominant Hand Only) processed for easy use in Python in IMU event tracking.'
			  ],
			  ["DICTIONARY ENTRIES IN DATABASE",
			   'README: Description of Pickle Database',
			   'UniqueID: Contains a participant and course number to uniquely identify each meal file in the original Database.',
			   'proc_data: Sensor readings of the dominant hand as reported in OREBA, processed to be standardized/normalized, gravity removed, and mirroring all left handed meals to right handed. Given as acc_x, acc_y, acc_z, then gyro_x, gyro_y, gyro_z (pitch roll yaw). Due to standardization, all values are unitless.',
			   'data_times: time [sec] from the start of the data, used to index answer.',
			   'handedness: list with an entry for each meal of whether the user was left or right hand dominant, and thus on which hand the data collection device was. All data has been mirrored to imitate the right hand.',
			   'bites_gt: list of lists, each containing a listed pair start time and end time (secs) of each intake moment.',
			   'bites_handedness_gt: list of lists, containing whether the corresponding ground truth bite was eaten with the left, right, or both hands.',
			  ]
			 ]


	OREBA_Cucumber={
		'README': README,
		'UniqueID': UniqueID,
		'proc_data': AllData,
		'data_times': AllDataTimes,
		'handedness': DomHand,
		'bites_gt': AllBites,
		'bites_handedness_gt': AllBiteHandedness
	}

	print("Finished Making Cucumber for Pickling")


	outfile = open("../Dnd_OneHandOreba.pkl",'wb')
	pkl.dump(OREBA_Cucumber,outfile)
	outfile.close()

	print("Finished Pickling OREBA Dataset")




# end of main()
