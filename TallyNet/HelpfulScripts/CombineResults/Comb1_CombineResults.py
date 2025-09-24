import sys






if __name__ == '__main__':
	
	FileList_filepath=sys.argv[1] # .txt file containing all file paths to find and combine results
	OutputFilename=sys.argv[2] # output .txt file for all compiled results
	
	# Constants to use based on the file output.
	NumHeaderLines = 0 # duplicate information to remove and test for consistency in files
	NumDataRows = 5 # the actual data results that are being condensed to a single file
	
	ValidHeaders_Flag = True # whether to check that headers have same formatting 
	
	
	# # # # # Collect All Files To Combine # # # # #
	FileList=[] # initialize empty list
	with open(FileList_filepath) as fpt:
		for currLine in fpt:
			FileList.append(currLine[0:-1]) #remove newline char
	# end of reading FileList_filepath
	
	
	# # # # # Grab Header Info From First File For Validation # # # # #
	with open(FileList[0]) as fpt:
		# grab all lines in a list without newline char
		#ValidationLines=[line[0:-1] for line in fpt] 
		ValidationLines=[line for line in fpt] 
	# end of open currFile
	
	if ValidHeaders_Flag:
		ValidHeader=ValidationLines[0:NumHeaderLines]
	#end of if
	
	output_fpt=open(OutputFilename,'w')
# 	Valid = ["{}\n".format(month) for month in months]
	output_fpt.writelines(ValidationLines)
		
	
	
	
	for currFile in FileList[1:]:
		with open(currFile) as fpt:
			# grab all lines in a list without newline char
			AllLines=[line for line in fpt] 
			# end of open currFile

			if ValidHeaders_Flag:
				ErrorFlag=0
				for i in range(NumHeaderLines):
					if AllLines[i] != ValidHeader[i]:
						print("!!! ERROR: INCONSISTENT HEADER INFO IN FILE \'{}\'".format(currFile))
						print("Skipping File...")
						ErrorFlag=1 # set ErrorFlag
				# end of for i loop
				if ErrorFlag==1:
					next # skip invalid file
			# end of if ValidHeaders_Flag

			output_fpt.writelines(AllLines[-NumDataRows:])
		
	# end of for currFile loop
	output_fpt.close()
	print("Finished combining all files.")
	