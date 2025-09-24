'''
File: GetResultFilenames.py
Author: James Jolly
Purpose: 
	Saves space in main functions which are needed to define and access
	filenames for results files of CTC reimplementation.
'''



def GetResultFilenames(DatasetFlag):
	'''
	Func: GetResultFilenames()
	Purpose: Output reimplementation CTC result filenames along the fold split used in training TallyNet   
		INPUTS
	DatasetFlag - (int)
			Which dataset to use: Can be one of the following values:
				1 = Dnd One Hand OREBA
				2 = Dnd Two Hand OREBA
				3 = Dnd Clemson
				4 = Dom One Hand OREBA
				5 = Dom Clemson
		OUTPUT
	ResultsFilenames  - (list of lists of strings), size: (5xN)
			List with 5 elements (for each of the 5 folds), each containing filenames for the specified dataset
			used for evaluation for that fold
	'''
	
	if DatasetFlag > 5:
		print("Invalid dataset flag in GetResultFilenames()... returning no files.")
		# return [[],[],[],[],[]] # either return empty set of 5 to match expected output size
		return [] # return empty set as error
	
	elif DatasetFlag == 1: # Dnd OHO
		ResultsFilenames = [
			# Fold 0
			[
				"OneHandOreba_1005_1_grm_std_uni.csv", "OneHandOreba_1044_1_grm_std_uni.csv", "OneHandOreba_1089_1_grm_std_uni.csv",
				"OneHandOreba_1011_1_grm_std_uni.csv", "OneHandOreba_1050_1_grm_std_uni.csv", "OneHandOreba_1094_1_grm_std_uni.csv",
				"OneHandOreba_1016_1_grm_std_uni.csv", "OneHandOreba_1055_1_grm_std_uni.csv", "OneHandOreba_1099_1_grm_std_uni.csv",
				"OneHandOreba_1021_1_grm_std_uni.csv", "OneHandOreba_1061_1_grm_std_uni.csv", "OneHandOreba_1104_1_grm_std_uni.csv",
				"OneHandOreba_1026_1_grm_std_uni.csv", "OneHandOreba_1072_1_grm_std_uni.csv", "OneHandOreba_1110_1_grm_std_uni.csv",
				"OneHandOreba_1031_1_grm_std_uni.csv", "OneHandOreba_1079_1_grm_std_uni.csv", "OneHandOreba_1116_1_grm_std_uni.csv",
				"OneHandOreba_1037_1_grm_std_uni.csv", "OneHandOreba_1084_1_grm_std_uni.csv"
			],
			
			# Fold 1
			[
				"OneHandOreba_1001_1_grm_std_uni.csv", "OneHandOreba_1039_1_grm_std_uni.csv", "OneHandOreba_1085_1_grm_std_uni.csv",
				"OneHandOreba_1006_1_grm_std_uni.csv", "OneHandOreba_1045_1_grm_std_uni.csv", "OneHandOreba_1090_1_grm_std_uni.csv",
				"OneHandOreba_1012_1_grm_std_uni.csv", "OneHandOreba_1051_1_grm_std_uni.csv", "OneHandOreba_1095_1_grm_std_uni.csv",
				"OneHandOreba_1017_1_grm_std_uni.csv", "OneHandOreba_1056_1_grm_std_uni.csv", "OneHandOreba_1100_1_grm_std_uni.csv",
				"OneHandOreba_1022_1_grm_std_uni.csv", "OneHandOreba_1063_1_grm_std_uni.csv", "OneHandOreba_1105_1_grm_std_uni.csv",
				"OneHandOreba_1027_1_grm_std_uni.csv", "OneHandOreba_1073_1_grm_std_uni.csv", "OneHandOreba_1111_1_grm_std_uni.csv",
				"OneHandOreba_1032_1_grm_std_uni.csv", "OneHandOreba_1080_1_grm_std_uni.csv"
			],
			
			# Fold 2
			[
				"OneHandOreba_1002_1_grm_std_uni.csv", "OneHandOreba_1040_1_grm_std_uni.csv", "OneHandOreba_1086_1_grm_std_uni.csv",
				"OneHandOreba_1007_1_grm_std_uni.csv", "OneHandOreba_1046_1_grm_std_uni.csv", "OneHandOreba_1091_1_grm_std_uni.csv",
				"OneHandOreba_1013_1_grm_std_uni.csv", "OneHandOreba_1052_1_grm_std_uni.csv", "OneHandOreba_1096_1_grm_std_uni.csv",
				"OneHandOreba_1018_1_grm_std_uni.csv", "OneHandOreba_1057_1_grm_std_uni.csv", "OneHandOreba_1101_1_grm_std_uni.csv",
				"OneHandOreba_1023_1_grm_std_uni.csv", "OneHandOreba_1064_1_grm_std_uni.csv", "OneHandOreba_1107_1_grm_std_uni.csv",
				"OneHandOreba_1028_1_grm_std_uni.csv", "OneHandOreba_1075_1_grm_std_uni.csv", "OneHandOreba_1112_1_grm_std_uni.csv",
				"OneHandOreba_1033_1_grm_std_uni.csv", "OneHandOreba_1081_1_grm_std_uni.csv"
			],
			
			# Fold 3
			[
				"OneHandOreba_1003_1_grm_std_uni.csv", "OneHandOreba_1041_1_grm_std_uni.csv", "OneHandOreba_1087_1_grm_std_uni.csv",
				"OneHandOreba_1008_1_grm_std_uni.csv", "OneHandOreba_1047_1_grm_std_uni.csv", "OneHandOreba_1092_1_grm_std_uni.csv",
				"OneHandOreba_1014_1_grm_std_uni.csv", "OneHandOreba_1053_1_grm_std_uni.csv", "OneHandOreba_1097_1_grm_std_uni.csv",
				"OneHandOreba_1019_1_grm_std_uni.csv", "OneHandOreba_1059_1_grm_std_uni.csv", "OneHandOreba_1102_1_grm_std_uni.csv",
				"OneHandOreba_1024_1_grm_std_uni.csv", "OneHandOreba_1067_1_grm_std_uni.csv", "OneHandOreba_1108_1_grm_std_uni.csv",
				"OneHandOreba_1029_1_grm_std_uni.csv", "OneHandOreba_1076_1_grm_std_uni.csv", "OneHandOreba_1113_1_grm_std_uni.csv",
				"OneHandOreba_1035_1_grm_std_uni.csv", "OneHandOreba_1082_1_grm_std_uni.csv"
			],
			
			# Fold 4
			[
				"OneHandOreba_1004_1_grm_std_uni.csv", "OneHandOreba_1043_1_grm_std_uni.csv", "OneHandOreba_1088_1_grm_std_uni.csv",
				"OneHandOreba_1010_1_grm_std_uni.csv", "OneHandOreba_1048_1_grm_std_uni.csv", "OneHandOreba_1093_1_grm_std_uni.csv",
				"OneHandOreba_1015_1_grm_std_uni.csv", "OneHandOreba_1054_1_grm_std_uni.csv", "OneHandOreba_1098_1_grm_std_uni.csv",
				"OneHandOreba_1020_1_grm_std_uni.csv", "OneHandOreba_1060_1_grm_std_uni.csv", "OneHandOreba_1103_1_grm_std_uni.csv",
				"OneHandOreba_1025_1_grm_std_uni.csv", "OneHandOreba_1068_1_grm_std_uni.csv", "OneHandOreba_1109_1_grm_std_uni.csv",
				"OneHandOreba_1030_1_grm_std_uni.csv", "OneHandOreba_1077_1_grm_std_uni.csv", "OneHandOreba_1115_1_grm_std_uni.csv",
				"OneHandOreba_1036_1_grm_std_uni.csv", "OneHandOreba_1083_1_grm_std_uni.csv"
			]
		]

	elif DatasetFlag == 2: # Dnd THO
		ResultsFilenames = [
			# Fold 0
			[
				'OREBA-DIS_1005_1_grm_std_uni.csv',  'OREBA-DIS_1011_1_grm_std_uni.csv',  	'OREBA-DIS_1016_1_grm_std_uni.csv',  
				'OREBA-DIS_1021_1_grm_std_uni.csv',  'OREBA-DIS_1026_1_grm_std_uni.csv',  	'OREBA-DIS_1031_1_grm_std_uni.csv',  
				'OREBA-DIS_1037_1_grm_std_uni.csv',  'OREBA-DIS_1044_1_grm_std_uni.csv',  	'OREBA-DIS_1050_1_grm_std_uni.csv',  
				'OREBA-DIS_1055_1_grm_std_uni.csv',	'OREBA-DIS_1061_1_grm_std_uni.csv',	'OREBA-DIS_1072_1_grm_std_uni.csv',
				'OREBA-DIS_1079_1_grm_std_uni.csv','OREBA-DIS_1084_1_grm_std_uni.csv',	'OREBA-DIS_1089_1_grm_std_uni.csv',
				'OREBA-DIS_1094_1_grm_std_uni.csv','OREBA-DIS_1099_1_grm_std_uni.csv',	'OREBA-DIS_1104_1_grm_std_uni.csv',
				'OREBA-DIS_1110_1_grm_std_uni.csv',	'OREBA-DIS_1116_1_grm_std_uni.csv'
			],
			
			# Fold 1
			[
				'OREBA-DIS_1001_1_grm_std_uni.csv',	'OREBA-DIS_1006_1_grm_std_uni.csv',	'OREBA-DIS_1012_1_grm_std_uni.csv',
				'OREBA-DIS_1017_1_grm_std_uni.csv',	'OREBA-DIS_1022_1_grm_std_uni.csv',	'OREBA-DIS_1027_1_grm_std_uni.csv',
				'OREBA-DIS_1032_1_grm_std_uni.csv',	'OREBA-DIS_1039_1_grm_std_uni.csv',	'OREBA-DIS_1045_1_grm_std_uni.csv',
				'OREBA-DIS_1051_1_grm_std_uni.csv',	'OREBA-DIS_1056_1_grm_std_uni.csv',	'OREBA-DIS_1063_1_grm_std_uni.csv',
				'OREBA-DIS_1073_1_grm_std_uni.csv',	'OREBA-DIS_1080_1_grm_std_uni.csv',	'OREBA-DIS_1085_1_grm_std_uni.csv',
				'OREBA-DIS_1090_1_grm_std_uni.csv',	'OREBA-DIS_1095_1_grm_std_uni.csv',	'OREBA-DIS_1100_1_grm_std_uni.csv',
				'OREBA-DIS_1105_1_grm_std_uni.csv',	'OREBA-DIS_1111_1_grm_std_uni.csv'
			],
			
			# Fold 2
			[
				'OREBA-DIS_1002_1_grm_std_uni.csv',	'OREBA-DIS_1007_1_grm_std_uni.csv',	'OREBA-DIS_1013_1_grm_std_uni.csv',
				'OREBA-DIS_1018_1_grm_std_uni.csv',	'OREBA-DIS_1023_1_grm_std_uni.csv',	'OREBA-DIS_1028_1_grm_std_uni.csv',
				'OREBA-DIS_1033_1_grm_std_uni.csv',	'OREBA-DIS_1040_1_grm_std_uni.csv',	'OREBA-DIS_1046_1_grm_std_uni.csv',
				'OREBA-DIS_1052_1_grm_std_uni.csv',	'OREBA-DIS_1057_1_grm_std_uni.csv',	'OREBA-DIS_1064_1_grm_std_uni.csv',
				'OREBA-DIS_1075_1_grm_std_uni.csv',	'OREBA-DIS_1081_1_grm_std_uni.csv',	'OREBA-DIS_1086_1_grm_std_uni.csv',
				'OREBA-DIS_1091_1_grm_std_uni.csv',	'OREBA-DIS_1096_1_grm_std_uni.csv',	'OREBA-DIS_1101_1_grm_std_uni.csv',
				'OREBA-DIS_1107_1_grm_std_uni.csv',	'OREBA-DIS_1112_1_grm_std_uni.csv'
			],
			
			# Fold 3
			[
				'OREBA-DIS_1003_1_grm_std_uni.csv',	'OREBA-DIS_1008_1_grm_std_uni.csv',	'OREBA-DIS_1014_1_grm_std_uni.csv',
				'OREBA-DIS_1019_1_grm_std_uni.csv',	'OREBA-DIS_1024_1_grm_std_uni.csv',	'OREBA-DIS_1029_1_grm_std_uni.csv',
				'OREBA-DIS_1035_1_grm_std_uni.csv',	'OREBA-DIS_1041_1_grm_std_uni.csv',	'OREBA-DIS_1047_1_grm_std_uni.csv',
				'OREBA-DIS_1053_1_grm_std_uni.csv',	'OREBA-DIS_1059_1_grm_std_uni.csv',	'OREBA-DIS_1067_1_grm_std_uni.csv',
				'OREBA-DIS_1076_1_grm_std_uni.csv',	'OREBA-DIS_1082_1_grm_std_uni.csv',	'OREBA-DIS_1087_1_grm_std_uni.csv',
				'OREBA-DIS_1092_1_grm_std_uni.csv',	'OREBA-DIS_1097_1_grm_std_uni.csv',	'OREBA-DIS_1102_1_grm_std_uni.csv',
				'OREBA-DIS_1108_1_grm_std_uni.csv',	'OREBA-DIS_1113_1_grm_std_uni.csv'
			],
			
			# Fold 4
			[
				'OREBA-DIS_1004_1_grm_std_uni.csv',	'OREBA-DIS_1010_1_grm_std_uni.csv',	'OREBA-DIS_1015_1_grm_std_uni.csv',
				'OREBA-DIS_1020_1_grm_std_uni.csv',	'OREBA-DIS_1025_1_grm_std_uni.csv',	'OREBA-DIS_1030_1_grm_std_uni.csv',
				'OREBA-DIS_1036_1_grm_std_uni.csv',	'OREBA-DIS_1043_1_grm_std_uni.csv',	'OREBA-DIS_1048_1_grm_std_uni.csv',
				'OREBA-DIS_1054_1_grm_std_uni.csv',	'OREBA-DIS_1060_1_grm_std_uni.csv',	'OREBA-DIS_1068_1_grm_std_uni.csv',
				'OREBA-DIS_1077_1_grm_std_uni.csv',	'OREBA-DIS_1083_1_grm_std_uni.csv',	'OREBA-DIS_1088_1_grm_std_uni.csv',
				'OREBA-DIS_1093_1_grm_std_uni.csv',	'OREBA-DIS_1098_1_grm_std_uni.csv',	'OREBA-DIS_1103_1_grm_std_uni.csv',
				'OREBA-DIS_1109_1_grm_std_uni.csv',	'OREBA-DIS_1115_1_grm_std_uni.csv'
			]
		]
		
	elif DatasetFlag == 3: # Dnd Clemson
		ResultsFilenames = [
			# FOLD 0 Files
			[
				'Clemson-DND_p007_c1_grm_std_uni.csv',	'Clemson-DND_p015_c1_grm_std_uni.csv',	'Clemson-DND_p019_c2_grm_std_uni.csv',
				'Clemson-DND_p024_c1_grm_std_uni.csv',	'Clemson-DND_p026_c3_grm_std_uni.csv',	'Clemson-DND_p028_c2_grm_std_uni.csv',
				'Clemson-DND_p034_c1_grm_std_uni.csv',	'Clemson-DND_p036_c1_grm_std_uni.csv',	'Clemson-DND_p039_c1_grm_std_uni.csv',
				'Clemson-DND_p045_c1_grm_std_uni.csv',	'Clemson-DND_p048_c1_grm_std_uni.csv',	'Clemson-DND_p052_c2_grm_std_uni.csv',
				'Clemson-DND_p055_c2_grm_std_uni.csv',	'Clemson-DND_p057_c3_grm_std_uni.csv',	'Clemson-DND_p060_c3_grm_std_uni.csv',
				'Clemson-DND_p062_c2_grm_std_uni.csv',	'Clemson-DND_p066_c1_grm_std_uni.csv',	'Clemson-DND_p068_c1_grm_std_uni.csv',
				'Clemson-DND_p070_c1_grm_std_uni.csv',	'Clemson-DND_p072_c1_grm_std_uni.csv',	'Clemson-DND_p077_c2_grm_std_uni.csv',
				'Clemson-DND_p079_c2_grm_std_uni.csv',	'Clemson-DND_p082_c2_grm_std_uni.csv',	'Clemson-DND_p085_c1_grm_std_uni.csv',
				'Clemson-DND_p088_c2_grm_std_uni.csv',	'Clemson-DND_p092_c1_grm_std_uni.csv',	'Clemson-DND_p098_c1_grm_std_uni.csv',
				'Clemson-DND_p101_c1_grm_std_uni.csv',	'Clemson-DND_p103_c2_grm_std_uni.csv',	'Clemson-DND_p107_c2_grm_std_uni.csv',
				'Clemson-DND_p110_c1_grm_std_uni.csv',	'Clemson-DND_p114_c1_grm_std_uni.csv',	'Clemson-DND_p116_c3_grm_std_uni.csv',
				'Clemson-DND_p118_c1_grm_std_uni.csv',	'Clemson-DND_p120_c2_grm_std_uni.csv',	'Clemson-DND_p123_c2_grm_std_uni.csv',
				'Clemson-DND_p130_c1_grm_std_uni.csv',	'Clemson-DND_p136_c1_grm_std_uni.csv',	'Clemson-DND_p140_c1_grm_std_uni.csv',
				'Clemson-DND_p144_c1_grm_std_uni.csv',	'Clemson-DND_p146_c3_grm_std_uni.csv',	'Clemson-DND_p151_c1_grm_std_uni.csv',
				'Clemson-DND_p157_c2_grm_std_uni.csv',	'Clemson-DND_p160_c2_grm_std_uni.csv',	'Clemson-DND_p165_c1_grm_std_uni.csv',
				'Clemson-DND_p169_c2_grm_std_uni.csv',	'Clemson-DND_p172_c2_grm_std_uni.csv',	'Clemson-DND_p174_c3_grm_std_uni.csv',
				'Clemson-DND_p176_c2_grm_std_uni.csv',	'Clemson-DND_p179_c1_grm_std_uni.csv',	'Clemson-DND_p180_c3_grm_std_uni.csv',
				'Clemson-DND_p184_c2_grm_std_uni.csv',	'Clemson-DND_p187_c2_grm_std_uni.csv',	'Clemson-DND_p190_c1_grm_std_uni.csv',
				'Clemson-DND_p194_c2_grm_std_uni.csv',	'Clemson-DND_p198_c1_grm_std_uni.csv',	'Clemson-DND_p204_c2_grm_std_uni.csv',
				'Clemson-DND_p207_c1_grm_std_uni.csv',	'Clemson-DND_p209_c1_grm_std_uni.csv',	'Clemson-DND_p217_c1_grm_std_uni.csv',
				'Clemson-DND_p218_c3_grm_std_uni.csv',	'Clemson-DND_p220_c2_grm_std_uni.csv',	'Clemson-DND_p229_c1_grm_std_uni.csv',
				'Clemson-DND_p231_c2_grm_std_uni.csv',	'Clemson-DND_p235_c1_grm_std_uni.csv',	'Clemson-DND_p241_c1_grm_std_uni.csv',
				'Clemson-DND_p244_c2_grm_std_uni.csv',	'Clemson-DND_p248_c1_grm_std_uni.csv',	'Clemson-DND_p253_c2_grm_std_uni.csv',
				'Clemson-DND_p260_c1_grm_std_uni.csv',	'Clemson-DND_p263_c1_grm_std_uni.csv',	'Clemson-DND_p265_c2_grm_std_uni.csv',
				'Clemson-DND_p268_c1_grm_std_uni.csv',	'Clemson-DND_p270_c2_grm_std_uni.csv',	'Clemson-DND_p271_c2_grm_std_uni.csv',
				'Clemson-DND_p273_c1_grm_std_uni.csv',	'Clemson-DND_p275_c2_grm_std_uni.csv',	'Clemson-DND_p277_c3_grm_std_uni.csv',
				'Clemson-DND_p279_c2_grm_std_uni.csv',	'Clemson-DND_p281_c3_grm_std_uni.csv',	'Clemson-DND_p285_c2_grm_std_uni.csv',
				'Clemson-DND_p291_c2_grm_std_uni.csv',	'Clemson-DND_p297_c2_grm_std_uni.csv',	'Clemson-DND_p311_c1_grm_std_uni.csv',
				'Clemson-DND_p315_c1_grm_std_uni.csv',	'Clemson-DND_p322_c1_grm_std_uni.csv',	'Clemson-DND_p326_c1_grm_std_uni.csv',
				'Clemson-DND_p331_c1_grm_std_uni.csv',	'Clemson-DND_p334_c1_grm_std_uni.csv',	'Clemson-DND_p337_c2_grm_std_uni.csv',
				'Clemson-DND_p343_c2_grm_std_uni.csv',	'Clemson-DND_p352_c2_grm_std_uni.csv',	'Clemson-DND_p361_c3_grm_std_uni.csv',
				'Clemson-DND_p372_c2_grm_std_uni.csv',	'Clemson-DND_p396_c1_grm_std_uni.csv',	'Clemson-DND_p401_c1_grm_std_uni.csv',
				'Clemson-DND_p410_c2_grm_std_uni.csv'
				],

			# FOLD 1 Files
			[
				'Clemson-DND_p005_c1_grm_std_uni.csv',	'Clemson-DND_p011_c1_grm_std_uni.csv',	'Clemson-DND_p016_c1_grm_std_uni.csv',
				'Clemson-DND_p020_c1_grm_std_uni.csv',	'Clemson-DND_p024_c2_grm_std_uni.csv',	'Clemson-DND_p027_c1_grm_std_uni.csv',
				'Clemson-DND_p029_c1_grm_std_uni.csv',	'Clemson-DND_p034_c2_grm_std_uni.csv',	'Clemson-DND_p036_c2_grm_std_uni.csv',
				'Clemson-DND_p042_c1_grm_std_uni.csv',	'Clemson-DND_p045_c2_grm_std_uni.csv',	'Clemson-DND_p050_c1_grm_std_uni.csv',
				'Clemson-DND_p053_c1_grm_std_uni.csv',	'Clemson-DND_p056_c1_grm_std_uni.csv',	'Clemson-DND_p058_c1_grm_std_uni.csv',
				'Clemson-DND_p061_c1_grm_std_uni.csv',	'Clemson-DND_p064_c1_grm_std_uni.csv',	'Clemson-DND_p066_c2_grm_std_uni.csv',
				'Clemson-DND_p068_c2_grm_std_uni.csv',	'Clemson-DND_p070_c2_grm_std_uni.csv',	'Clemson-DND_p074_c1_grm_std_uni.csv',
				'Clemson-DND_p077_c3_grm_std_uni.csv',	'Clemson-DND_p080_c1_grm_std_uni.csv',	'Clemson-DND_p083_c1_grm_std_uni.csv',
				'Clemson-DND_p086_c1_grm_std_uni.csv',	'Clemson-DND_p089_c1_grm_std_uni.csv',	'Clemson-DND_p093_c1_grm_std_uni.csv',
				'Clemson-DND_p098_c2_grm_std_uni.csv',	'Clemson-DND_p101_c2_grm_std_uni.csv',	'Clemson-DND_p104_c1_grm_std_uni.csv',
				'Clemson-DND_p108_c1_grm_std_uni.csv',	'Clemson-DND_p111_c1_grm_std_uni.csv',	'Clemson-DND_p114_c2_grm_std_uni.csv',
				'Clemson-DND_p117_c1_grm_std_uni.csv',	'Clemson-DND_p118_c2_grm_std_uni.csv',	'Clemson-DND_p121_c1_grm_std_uni.csv',
				'Clemson-DND_p125_c1_grm_std_uni.csv',	'Clemson-DND_p131_c1_grm_std_uni.csv',	'Clemson-DND_p137_c1_grm_std_uni.csv',
				'Clemson-DND_p142_c1_grm_std_uni.csv',	'Clemson-DND_p144_c2_grm_std_uni.csv',	'Clemson-DND_p148_c1_grm_std_uni.csv',
				'Clemson-DND_p151_c2_grm_std_uni.csv',	'Clemson-DND_p158_c1_grm_std_uni.csv',	'Clemson-DND_p161_c1_grm_std_uni.csv',
				'Clemson-DND_p165_c2_grm_std_uni.csv',	'Clemson-DND_p170_c1_grm_std_uni.csv',	'Clemson-DND_p173_c1_grm_std_uni.csv',
				'Clemson-DND_p175_c1_grm_std_uni.csv',	'Clemson-DND_p177_c1_grm_std_uni.csv',	'Clemson-DND_p179_c2_grm_std_uni.csv',
				'Clemson-DND_p181_c1_grm_std_uni.csv',	'Clemson-DND_p185_c1_grm_std_uni.csv',	'Clemson-DND_p187_c3_grm_std_uni.csv',
				'Clemson-DND_p190_c2_grm_std_uni.csv',	'Clemson-DND_p194_c3_grm_std_uni.csv',	'Clemson-DND_p199_c1_grm_std_uni.csv',
				'Clemson-DND_p204_c3_grm_std_uni.csv',	'Clemson-DND_p207_c2_grm_std_uni.csv',	'Clemson-DND_p209_c2_grm_std_uni.csv',
				'Clemson-DND_p217_c2_grm_std_uni.csv',	'Clemson-DND_p218_c4_grm_std_uni.csv',	'Clemson-DND_p221_c1_grm_std_uni.csv',
				'Clemson-DND_p229_c2_grm_std_uni.csv',	'Clemson-DND_p232_c1_grm_std_uni.csv',	'Clemson-DND_p236_c1_grm_std_uni.csv',
				'Clemson-DND_p241_c2_grm_std_uni.csv',	'Clemson-DND_p245_c1_grm_std_uni.csv',	'Clemson-DND_p251_c2_grm_std_uni.csv',
				'Clemson-DND_p256_c1_grm_std_uni.csv',	'Clemson-DND_p260_c2_grm_std_uni.csv',	'Clemson-DND_p263_c2_grm_std_uni.csv',
				'Clemson-DND_p266_c1_grm_std_uni.csv',	'Clemson-DND_p268_c2_grm_std_uni.csv',	'Clemson-DND_p270_c3_grm_std_uni.csv',
				'Clemson-DND_p271_c3_grm_std_uni.csv',	'Clemson-DND_p273_c2_grm_std_uni.csv',	'Clemson-DND_p276_c1_grm_std_uni.csv',
				'Clemson-DND_p278_c1_grm_std_uni.csv',	'Clemson-DND_p279_c3_grm_std_uni.csv',	'Clemson-DND_p282_c1_grm_std_uni.csv',
				'Clemson-DND_p289_c2_grm_std_uni.csv',	'Clemson-DND_p292_c1_grm_std_uni.csv',	'Clemson-DND_p298_c1_grm_std_uni.csv',
				'Clemson-DND_p311_c2_grm_std_uni.csv',	'Clemson-DND_p315_c2_grm_std_uni.csv',	'Clemson-DND_p322_c2_grm_std_uni.csv',
				'Clemson-DND_p326_c2_grm_std_uni.csv',	'Clemson-DND_p331_c2_grm_std_uni.csv',	'Clemson-DND_p334_c2_grm_std_uni.csv',
				'Clemson-DND_p338_c1_grm_std_uni.csv',	'Clemson-DND_p343_c3_grm_std_uni.csv',	'Clemson-DND_p352_c3_grm_std_uni.csv',
				'Clemson-DND_p368_c1_grm_std_uni.csv',	'Clemson-DND_p377_c1_grm_std_uni.csv',	'Clemson-DND_p396_c2_grm_std_uni.csv',
				'Clemson-DND_p401_c2_grm_std_uni.csv',	'Clemson-DND_p411_c1_grm_std_uni.csv'
			],

			# FOLD 2 Files
			[
				'Clemson-DND_p005_c2_grm_std_uni.csv',	'Clemson-DND_p011_c2_grm_std_uni.csv',	'Clemson-DND_p016_c2_grm_std_uni.csv',
				'Clemson-DND_p021_c2_grm_std_uni.csv',	'Clemson-DND_p025_c1_grm_std_uni.csv',	'Clemson-DND_p027_c2_grm_std_uni.csv',
				'Clemson-DND_p030_c1_grm_std_uni.csv',	'Clemson-DND_p034_c3_grm_std_uni.csv',	'Clemson-DND_p037_c1_grm_std_uni.csv',
				'Clemson-DND_p042_c2_grm_std_uni.csv',	'Clemson-DND_p045_c3_grm_std_uni.csv',	'Clemson-DND_p051_c1_grm_std_uni.csv',
				'Clemson-DND_p054_c1_grm_std_uni.csv',	'Clemson-DND_p056_c2_grm_std_uni.csv',	'Clemson-DND_p059_c1_grm_std_uni.csv',
				'Clemson-DND_p061_c2_grm_std_uni.csv',	'Clemson-DND_p064_c2_grm_std_uni.csv',	'Clemson-DND_p066_c3_grm_std_uni.csv',
				'Clemson-DND_p069_c1_grm_std_uni.csv',	'Clemson-DND_p070_c3_grm_std_uni.csv',	'Clemson-DND_p074_c2_grm_std_uni.csv',
				'Clemson-DND_p078_c1_grm_std_uni.csv',	'Clemson-DND_p080_c2_grm_std_uni.csv',	'Clemson-DND_p083_c2_grm_std_uni.csv',
				'Clemson-DND_p087_c1_grm_std_uni.csv',	'Clemson-DND_p090_c1_grm_std_uni.csv',	'Clemson-DND_p093_c2_grm_std_uni.csv',
				'Clemson-DND_p099_c1_grm_std_uni.csv',	'Clemson-DND_p102_c1_grm_std_uni.csv',	'Clemson-DND_p105_c1_grm_std_uni.csv',
				'Clemson-DND_p108_c2_grm_std_uni.csv',	'Clemson-DND_p111_c2_grm_std_uni.csv',	'Clemson-DND_p115_c1_grm_std_uni.csv',
				'Clemson-DND_p117_c2_grm_std_uni.csv',	'Clemson-DND_p119_c1_grm_std_uni.csv',	'Clemson-DND_p121_c2_grm_std_uni.csv',
				'Clemson-DND_p125_c2_grm_std_uni.csv',	'Clemson-DND_p132_c1_grm_std_uni.csv',	'Clemson-DND_p137_c2_grm_std_uni.csv',
				'Clemson-DND_p142_c2_grm_std_uni.csv',	'Clemson-DND_p145_c1_grm_std_uni.csv',	'Clemson-DND_p148_c2_grm_std_uni.csv',
				'Clemson-DND_p153_c1_grm_std_uni.csv',	'Clemson-DND_p158_c2_grm_std_uni.csv',	'Clemson-DND_p161_c2_grm_std_uni.csv',
				'Clemson-DND_p166_c1_grm_std_uni.csv',	'Clemson-DND_p171_c1_grm_std_uni.csv',	'Clemson-DND_p173_c2_grm_std_uni.csv',
				'Clemson-DND_p175_c2_grm_std_uni.csv',	'Clemson-DND_p177_c2_grm_std_uni.csv',	'Clemson-DND_p179_c3_grm_std_uni.csv',
				'Clemson-DND_p182_c1_grm_std_uni.csv',	'Clemson-DND_p186_c1_grm_std_uni.csv',	'Clemson-DND_p188_c1_grm_std_uni.csv',
				'Clemson-DND_p192_c1_grm_std_uni.csv',	'Clemson-DND_p195_c1_grm_std_uni.csv',	'Clemson-DND_p201_c1_grm_std_uni.csv',
				'Clemson-DND_p205_c1_grm_std_uni.csv',	'Clemson-DND_p207_c3_grm_std_uni.csv',	'Clemson-DND_p215_c1_grm_std_uni.csv',
				'Clemson-DND_p217_c3_grm_std_uni.csv',	'Clemson-DND_p219_c1_grm_std_uni.csv',	'Clemson-DND_p224_c1_grm_std_uni.csv',
				'Clemson-DND_p230_c1_grm_std_uni.csv',	'Clemson-DND_p233_c1_grm_std_uni.csv',	'Clemson-DND_p236_c2_grm_std_uni.csv',
				'Clemson-DND_p242_c1_grm_std_uni.csv',	'Clemson-DND_p246_c1_grm_std_uni.csv',	'Clemson-DND_p252_c1_grm_std_uni.csv',
				'Clemson-DND_p257_c1_grm_std_uni.csv',	'Clemson-DND_p262_c1_grm_std_uni.csv',	'Clemson-DND_p264_c1_grm_std_uni.csv',
				'Clemson-DND_p266_c2_grm_std_uni.csv',	'Clemson-DND_p269_c1_grm_std_uni.csv',	'Clemson-DND_p270_c4_grm_std_uni.csv',
				'Clemson-DND_p272_c1_grm_std_uni.csv',	'Clemson-DND_p274_c1_grm_std_uni.csv',	'Clemson-DND_p276_c2_grm_std_uni.csv',
				'Clemson-DND_p278_c2_grm_std_uni.csv',	'Clemson-DND_p280_c1_grm_std_uni.csv',	'Clemson-DND_p283_c1_grm_std_uni.csv',
				'Clemson-DND_p290_c1_grm_std_uni.csv',	'Clemson-DND_p293_c1_grm_std_uni.csv',	'Clemson-DND_p298_c2_grm_std_uni.csv',
				'Clemson-DND_p311_c3_grm_std_uni.csv',	'Clemson-DND_p318_c1_grm_std_uni.csv',	'Clemson-DND_p322_c3_grm_std_uni.csv',
				'Clemson-DND_p329_c1_grm_std_uni.csv',	'Clemson-DND_p332_c1_grm_std_uni.csv',	'Clemson-DND_p336_c1_grm_std_uni.csv',
				'Clemson-DND_p338_c2_grm_std_uni.csv',	'Clemson-DND_p343_c4_grm_std_uni.csv',	'Clemson-DND_p353_c1_grm_std_uni.csv',
				'Clemson-DND_p368_c2_grm_std_uni.csv',	'Clemson-DND_p384_c2_grm_std_uni.csv',	'Clemson-DND_p396_c3_grm_std_uni.csv',
				'Clemson-DND_p406_c1_grm_std_uni.csv',	'Clemson-DND_p411_c2_grm_std_uni.csv'
			],

			# FOLD 3 Files
			[
				'Clemson-DND_p006_c1_grm_std_uni.csv',	'Clemson-DND_p012_c2_grm_std_uni.csv',	'Clemson-DND_p017_c2_grm_std_uni.csv',
				'Clemson-DND_p022_c1_grm_std_uni.csv',	'Clemson-DND_p026_c1_grm_std_uni.csv',	'Clemson-DND_p027_c3_grm_std_uni.csv',
				'Clemson-DND_p031_c1_grm_std_uni.csv',	'Clemson-DND_p035_c1_grm_std_uni.csv',	'Clemson-DND_p037_c2_grm_std_uni.csv',
				'Clemson-DND_p043_c1_grm_std_uni.csv',	'Clemson-DND_p046_c1_grm_std_uni.csv',	'Clemson-DND_p051_c2_grm_std_uni.csv',
				'Clemson-DND_p054_c2_grm_std_uni.csv',	'Clemson-DND_p057_c1_grm_std_uni.csv',	'Clemson-DND_p060_c1_grm_std_uni.csv',
				'Clemson-DND_p061_c3_grm_std_uni.csv',	'Clemson-DND_p065_c1_grm_std_uni.csv',	'Clemson-DND_p067_c1_grm_std_uni.csv',
				'Clemson-DND_p069_c3_grm_std_uni.csv',	'Clemson-DND_p071_c1_grm_std_uni.csv',	'Clemson-DND_p075_c1_grm_std_uni.csv',
				'Clemson-DND_p078_c2_grm_std_uni.csv',	'Clemson-DND_p081_c1_grm_std_uni.csv',	'Clemson-DND_p084_c1_grm_std_uni.csv',
				'Clemson-DND_p087_c2_grm_std_uni.csv',	'Clemson-DND_p090_c2_grm_std_uni.csv',	'Clemson-DND_p095_c1_grm_std_uni.csv',
				'Clemson-DND_p099_c2_grm_std_uni.csv',	'Clemson-DND_p102_c2_grm_std_uni.csv',	'Clemson-DND_p106_c1_grm_std_uni.csv',
				'Clemson-DND_p109_c1_grm_std_uni.csv',	'Clemson-DND_p113_c1_grm_std_uni.csv',	'Clemson-DND_p115_c2_grm_std_uni.csv',
				'Clemson-DND_p117_c3_grm_std_uni.csv',	'Clemson-DND_p119_c2_grm_std_uni.csv',	'Clemson-DND_p122_c1_grm_std_uni.csv',
				'Clemson-DND_p129_c1_grm_std_uni.csv',	'Clemson-DND_p132_c2_grm_std_uni.csv',	'Clemson-DND_p138_c1_grm_std_uni.csv',
				'Clemson-DND_p143_c1_grm_std_uni.csv',	'Clemson-DND_p146_c1_grm_std_uni.csv',	'Clemson-DND_p150_c1_grm_std_uni.csv',
				'Clemson-DND_p154_c1_grm_std_uni.csv',	'Clemson-DND_p159_c1_grm_std_uni.csv',	'Clemson-DND_p162_c1_grm_std_uni.csv',
				'Clemson-DND_p166_c2_grm_std_uni.csv',	'Clemson-DND_p171_c2_grm_std_uni.csv',	'Clemson-DND_p174_c1_grm_std_uni.csv',
				'Clemson-DND_p175_c3_grm_std_uni.csv',	'Clemson-DND_p178_c1_grm_std_uni.csv',	'Clemson-DND_p180_c1_grm_std_uni.csv',
				'Clemson-DND_p182_c2_grm_std_uni.csv',	'Clemson-DND_p186_c2_grm_std_uni.csv',	'Clemson-DND_p188_c2_grm_std_uni.csv',
				'Clemson-DND_p192_c2_grm_std_uni.csv',	'Clemson-DND_p195_c2_grm_std_uni.csv',	'Clemson-DND_p202_c1_grm_std_uni.csv',
				'Clemson-DND_p205_c2_grm_std_uni.csv',	'Clemson-DND_p208_c1_grm_std_uni.csv',	'Clemson-DND_p215_c2_grm_std_uni.csv',
				'Clemson-DND_p218_c1_grm_std_uni.csv',	'Clemson-DND_p219_c2_grm_std_uni.csv',	'Clemson-DND_p226_c1_grm_std_uni.csv',
				'Clemson-DND_p230_c2_grm_std_uni.csv',	'Clemson-DND_p234_c1_grm_std_uni.csv',	'Clemson-DND_p237_c1_grm_std_uni.csv',
				'Clemson-DND_p242_c2_grm_std_uni.csv',	'Clemson-DND_p247_c1_grm_std_uni.csv',	'Clemson-DND_p252_c2_grm_std_uni.csv',
				'Clemson-DND_p257_c2_grm_std_uni.csv',	'Clemson-DND_p262_c2_grm_std_uni.csv',	'Clemson-DND_p264_c2_grm_std_uni.csv',
				'Clemson-DND_p267_c1_grm_std_uni.csv',	'Clemson-DND_p269_c2_grm_std_uni.csv',	'Clemson-DND_p270_c5_grm_std_uni.csv',
				'Clemson-DND_p272_c2_grm_std_uni.csv',	'Clemson-DND_p274_c2_grm_std_uni.csv',	'Clemson-DND_p277_c1_grm_std_uni.csv',
				'Clemson-DND_p278_c3_grm_std_uni.csv',	'Clemson-DND_p280_c2_grm_std_uni.csv',	'Clemson-DND_p284_c1_grm_std_uni.csv',
				'Clemson-DND_p290_c2_grm_std_uni.csv',	'Clemson-DND_p293_c2_grm_std_uni.csv',	'Clemson-DND_p309_c1_grm_std_uni.csv',
				'Clemson-DND_p312_c1_grm_std_uni.csv',	'Clemson-DND_p320_c1_grm_std_uni.csv',	'Clemson-DND_p324_c1_grm_std_uni.csv',
				'Clemson-DND_p329_c2_grm_std_uni.csv',	'Clemson-DND_p332_c2_grm_std_uni.csv',	'Clemson-DND_p336_c2_grm_std_uni.csv',
				'Clemson-DND_p341_c1_grm_std_uni.csv',	'Clemson-DND_p347_c1_grm_std_uni.csv',	'Clemson-DND_p353_c2_grm_std_uni.csv',
				'Clemson-DND_p368_c3_grm_std_uni.csv',	'Clemson-DND_p392_c1_grm_std_uni.csv',	'Clemson-DND_p397_c1_grm_std_uni.csv',
				'Clemson-DND_p406_c2_grm_std_uni.csv',	'Clemson-DND_p413_c1_grm_std_uni.csv'
			],

			# FOLD 4 Files
			[
				'Clemson-DND_p006_c2_grm_std_uni.csv',	'Clemson-DND_p013_c1_grm_std_uni.csv',	'Clemson-DND_p019_c1_grm_std_uni.csv',
				'Clemson-DND_p023_c1_grm_std_uni.csv',	'Clemson-DND_p026_c2_grm_std_uni.csv',	'Clemson-DND_p028_c1_grm_std_uni.csv',
				'Clemson-DND_p033_c1_grm_std_uni.csv',	'Clemson-DND_p035_c2_grm_std_uni.csv',	'Clemson-DND_p038_c1_grm_std_uni.csv',
				'Clemson-DND_p044_c1_grm_std_uni.csv',	'Clemson-DND_p047_c2_grm_std_uni.csv',	'Clemson-DND_p052_c1_grm_std_uni.csv',
				'Clemson-DND_p055_c1_grm_std_uni.csv',	'Clemson-DND_p057_c2_grm_std_uni.csv',	'Clemson-DND_p060_c2_grm_std_uni.csv',
				'Clemson-DND_p062_c1_grm_std_uni.csv',	'Clemson-DND_p065_c2_grm_std_uni.csv',	'Clemson-DND_p067_c2_grm_std_uni.csv',
				'Clemson-DND_p069_c4_grm_std_uni.csv',	'Clemson-DND_p071_c2_grm_std_uni.csv',	'Clemson-DND_p077_c1_grm_std_uni.csv',
				'Clemson-DND_p079_c1_grm_std_uni.csv',	'Clemson-DND_p082_c1_grm_std_uni.csv',	'Clemson-DND_p084_c2_grm_std_uni.csv',
				'Clemson-DND_p088_c1_grm_std_uni.csv',	'Clemson-DND_p091_c1_grm_std_uni.csv',	'Clemson-DND_p096_c1_grm_std_uni.csv',
				'Clemson-DND_p100_c1_grm_std_uni.csv',	'Clemson-DND_p103_c1_grm_std_uni.csv',	'Clemson-DND_p107_c1_grm_std_uni.csv',
				'Clemson-DND_p109_c2_grm_std_uni.csv',	'Clemson-DND_p113_c2_grm_std_uni.csv',	'Clemson-DND_p116_c2_grm_std_uni.csv',
				'Clemson-DND_p117_c4_grm_std_uni.csv',	'Clemson-DND_p120_c1_grm_std_uni.csv',	'Clemson-DND_p122_c2_grm_std_uni.csv',
				'Clemson-DND_p129_c2_grm_std_uni.csv',	'Clemson-DND_p133_c1_grm_std_uni.csv',	'Clemson-DND_p139_c1_grm_std_uni.csv',
				'Clemson-DND_p143_c2_grm_std_uni.csv',	'Clemson-DND_p146_c2_grm_std_uni.csv',	'Clemson-DND_p150_c2_grm_std_uni.csv',
				'Clemson-DND_p157_c1_grm_std_uni.csv',	'Clemson-DND_p160_c1_grm_std_uni.csv',	'Clemson-DND_p164_c1_grm_std_uni.csv',
				'Clemson-DND_p169_c1_grm_std_uni.csv',	'Clemson-DND_p172_c1_grm_std_uni.csv',	'Clemson-DND_p174_c2_grm_std_uni.csv',
				'Clemson-DND_p176_c1_grm_std_uni.csv',	'Clemson-DND_p178_c2_grm_std_uni.csv',	'Clemson-DND_p180_c2_grm_std_uni.csv',
				'Clemson-DND_p184_c1_grm_std_uni.csv',	'Clemson-DND_p187_c1_grm_std_uni.csv',	'Clemson-DND_p189_c1_grm_std_uni.csv',
				'Clemson-DND_p194_c1_grm_std_uni.csv',	'Clemson-DND_p195_c3_grm_std_uni.csv',	'Clemson-DND_p204_c1_grm_std_uni.csv',
				'Clemson-DND_p206_c1_grm_std_uni.csv',	'Clemson-DND_p208_c2_grm_std_uni.csv',	'Clemson-DND_p215_c3_grm_std_uni.csv',
				'Clemson-DND_p218_c2_grm_std_uni.csv',	'Clemson-DND_p220_c1_grm_std_uni.csv',	'Clemson-DND_p226_c2_grm_std_uni.csv',
				'Clemson-DND_p231_c1_grm_std_uni.csv',	'Clemson-DND_p234_c2_grm_std_uni.csv',	'Clemson-DND_p237_c2_grm_std_uni.csv',
				'Clemson-DND_p244_c1_grm_std_uni.csv',	'Clemson-DND_p247_c2_grm_std_uni.csv',	'Clemson-DND_p253_c1_grm_std_uni.csv',
				'Clemson-DND_p259_c1_grm_std_uni.csv',	'Clemson-DND_p262_c3_grm_std_uni.csv',	'Clemson-DND_p265_c1_grm_std_uni.csv',
				'Clemson-DND_p267_c2_grm_std_uni.csv',	'Clemson-DND_p270_c1_grm_std_uni.csv',	'Clemson-DND_p271_c1_grm_std_uni.csv',
				'Clemson-DND_p272_c3_grm_std_uni.csv',	'Clemson-DND_p275_c1_grm_std_uni.csv',	'Clemson-DND_p277_c2_grm_std_uni.csv',
				'Clemson-DND_p279_c1_grm_std_uni.csv',	'Clemson-DND_p281_c1_grm_std_uni.csv',	'Clemson-DND_p285_c1_grm_std_uni.csv',
				'Clemson-DND_p291_c1_grm_std_uni.csv',	'Clemson-DND_p297_c1_grm_std_uni.csv',	'Clemson-DND_p309_c2_grm_std_uni.csv',
				'Clemson-DND_p312_c2_grm_std_uni.csv',	'Clemson-DND_p320_c2_grm_std_uni.csv',	'Clemson-DND_p324_c2_grm_std_uni.csv',
				'Clemson-DND_p329_c3_grm_std_uni.csv',	'Clemson-DND_p332_c3_grm_std_uni.csv',	'Clemson-DND_p337_c1_grm_std_uni.csv',
				'Clemson-DND_p343_c1_grm_std_uni.csv',	'Clemson-DND_p352_c1_grm_std_uni.csv',	'Clemson-DND_p353_c3_grm_std_uni.csv',
				'Clemson-DND_p372_c1_grm_std_uni.csv',	'Clemson-DND_p392_c2_grm_std_uni.csv',	'Clemson-DND_p397_c2_grm_std_uni.csv',
				'Clemson-DND_p410_c1_grm_std_uni.csv'
			]
		]
		
	elif DatasetFlag == 4: # Dom OHO
		ResultsFilenames = [
			# Fold 0
			[
				"OneHandOrebaV3_1005_1_grm_std_uni.csv",  "OneHandOrebaV3_1061_1_grm_std_uni.csv",
				"OneHandOrebaV3_1011_1_grm_std_uni.csv",  "OneHandOrebaV3_1072_1_grm_std_uni.csv",
				"OneHandOrebaV3_1016_1_grm_std_uni.csv",  "OneHandOrebaV3_1079_1_grm_std_uni.csv",
				"OneHandOrebaV3_1021_1_grm_std_uni.csv",  "OneHandOrebaV3_1084_1_grm_std_uni.csv",
				"OneHandOrebaV3_1026_1_grm_std_uni.csv",  "OneHandOrebaV3_1089_1_grm_std_uni.csv",
				"OneHandOrebaV3_1031_1_grm_std_uni.csv",  "OneHandOrebaV3_1094_1_grm_std_uni.csv",
				"OneHandOrebaV3_1037_1_grm_std_uni.csv",  "OneHandOrebaV3_1099_1_grm_std_uni.csv",
				"OneHandOrebaV3_1044_1_grm_std_uni.csv",  "OneHandOrebaV3_1104_1_grm_std_uni.csv",
				"OneHandOrebaV3_1050_1_grm_std_uni.csv",  "OneHandOrebaV3_1110_1_grm_std_uni.csv",
				"OneHandOrebaV3_1055_1_grm_std_uni.csv",  "OneHandOrebaV3_1116_1_grm_std_uni.csv"
			],
			
			# Fold 1
			[
				"OneHandOrebaV3_1001_1_grm_std_uni.csv",  "OneHandOrebaV3_1056_1_grm_std_uni.csv",
				"OneHandOrebaV3_1006_1_grm_std_uni.csv",  "OneHandOrebaV3_1063_1_grm_std_uni.csv",
				"OneHandOrebaV3_1012_1_grm_std_uni.csv",  "OneHandOrebaV3_1073_1_grm_std_uni.csv",
				"OneHandOrebaV3_1017_1_grm_std_uni.csv",  "OneHandOrebaV3_1080_1_grm_std_uni.csv",
				"OneHandOrebaV3_1022_1_grm_std_uni.csv",  "OneHandOrebaV3_1085_1_grm_std_uni.csv",
				"OneHandOrebaV3_1027_1_grm_std_uni.csv",  "OneHandOrebaV3_1090_1_grm_std_uni.csv",
				"OneHandOrebaV3_1032_1_grm_std_uni.csv",  "OneHandOrebaV3_1095_1_grm_std_uni.csv",
				"OneHandOrebaV3_1039_1_grm_std_uni.csv",  "OneHandOrebaV3_1100_1_grm_std_uni.csv",
				"OneHandOrebaV3_1045_1_grm_std_uni.csv",  "OneHandOrebaV3_1105_1_grm_std_uni.csv",
				"OneHandOrebaV3_1051_1_grm_std_uni.csv",  "OneHandOrebaV3_1111_1_grm_std_uni.csv"
			],
			
			# Fold 2
			[
				"OneHandOrebaV3_1002_1_grm_std_uni.csv",  "OneHandOrebaV3_1057_1_grm_std_uni.csv",
				"OneHandOrebaV3_1007_1_grm_std_uni.csv",  "OneHandOrebaV3_1064_1_grm_std_uni.csv",
				"OneHandOrebaV3_1013_1_grm_std_uni.csv",  "OneHandOrebaV3_1075_1_grm_std_uni.csv",
				"OneHandOrebaV3_1018_1_grm_std_uni.csv",  "OneHandOrebaV3_1081_1_grm_std_uni.csv",
				"OneHandOrebaV3_1023_1_grm_std_uni.csv",  "OneHandOrebaV3_1086_1_grm_std_uni.csv",
				"OneHandOrebaV3_1028_1_grm_std_uni.csv",  "OneHandOrebaV3_1091_1_grm_std_uni.csv",
				"OneHandOrebaV3_1033_1_grm_std_uni.csv",  "OneHandOrebaV3_1096_1_grm_std_uni.csv",
				"OneHandOrebaV3_1040_1_grm_std_uni.csv",  "OneHandOrebaV3_1101_1_grm_std_uni.csv",
				"OneHandOrebaV3_1046_1_grm_std_uni.csv",  "OneHandOrebaV3_1107_1_grm_std_uni.csv",
				"OneHandOrebaV3_1052_1_grm_std_uni.csv",  "OneHandOrebaV3_1112_1_grm_std_uni.csv"
			],
			
			# Fold 3
			[
				"OneHandOrebaV3_1003_1_grm_std_uni.csv",  "OneHandOrebaV3_1059_1_grm_std_uni.csv",
				"OneHandOrebaV3_1008_1_grm_std_uni.csv",  "OneHandOrebaV3_1067_1_grm_std_uni.csv",
				"OneHandOrebaV3_1014_1_grm_std_uni.csv",  "OneHandOrebaV3_1076_1_grm_std_uni.csv",
				"OneHandOrebaV3_1019_1_grm_std_uni.csv",  "OneHandOrebaV3_1082_1_grm_std_uni.csv",
				"OneHandOrebaV3_1024_1_grm_std_uni.csv",  "OneHandOrebaV3_1087_1_grm_std_uni.csv",
				"OneHandOrebaV3_1029_1_grm_std_uni.csv",  "OneHandOrebaV3_1092_1_grm_std_uni.csv",
				"OneHandOrebaV3_1035_1_grm_std_uni.csv",  "OneHandOrebaV3_1097_1_grm_std_uni.csv",
				"OneHandOrebaV3_1041_1_grm_std_uni.csv",  "OneHandOrebaV3_1102_1_grm_std_uni.csv",
				"OneHandOrebaV3_1047_1_grm_std_uni.csv",  "OneHandOrebaV3_1108_1_grm_std_uni.csv",
				"OneHandOrebaV3_1053_1_grm_std_uni.csv",  "OneHandOrebaV3_1113_1_grm_std_uni.csv"
			],
			
			# Fold 4
			[
				"OneHandOrebaV3_1004_1_grm_std_uni.csv",  "OneHandOrebaV3_1060_1_grm_std_uni.csv",
				"OneHandOrebaV3_1010_1_grm_std_uni.csv",  "OneHandOrebaV3_1068_1_grm_std_uni.csv",
				"OneHandOrebaV3_1015_1_grm_std_uni.csv",  "OneHandOrebaV3_1077_1_grm_std_uni.csv",
				"OneHandOrebaV3_1020_1_grm_std_uni.csv",  "OneHandOrebaV3_1083_1_grm_std_uni.csv",
				"OneHandOrebaV3_1025_1_grm_std_uni.csv",  "OneHandOrebaV3_1088_1_grm_std_uni.csv",
				"OneHandOrebaV3_1030_1_grm_std_uni.csv",  "OneHandOrebaV3_1093_1_grm_std_uni.csv",
				"OneHandOrebaV3_1036_1_grm_std_uni.csv",  "OneHandOrebaV3_1098_1_grm_std_uni.csv",
				"OneHandOrebaV3_1043_1_grm_std_uni.csv",  "OneHandOrebaV3_1103_1_grm_std_uni.csv",
				"OneHandOrebaV3_1048_1_grm_std_uni.csv",  "OneHandOrebaV3_1109_1_grm_std_uni.csv",
				"OneHandOrebaV3_1054_1_grm_std_uni.csv",  "OneHandOrebaV3_1115_1_grm_std_uni.csv"
			]
		]

		
	elif DatasetFlag == 5: # Dom Clemson
		ResultsFilenames = [
			# FOLD 0 Files
			[
				'Clemson_p007_c1_grm_std_uni.csv',	'Clemson_p015_c1_grm_std_uni.csv',	'Clemson_p019_c2_grm_std_uni.csv',
				'Clemson_p024_c1_grm_std_uni.csv',	'Clemson_p026_c3_grm_std_uni.csv',	'Clemson_p028_c2_grm_std_uni.csv',
				'Clemson_p034_c1_grm_std_uni.csv',	'Clemson_p036_c1_grm_std_uni.csv',	'Clemson_p039_c1_grm_std_uni.csv',
				'Clemson_p045_c1_grm_std_uni.csv',	'Clemson_p048_c1_grm_std_uni.csv',	'Clemson_p052_c2_grm_std_uni.csv',
				'Clemson_p055_c2_grm_std_uni.csv',	'Clemson_p057_c3_grm_std_uni.csv',	'Clemson_p060_c3_grm_std_uni.csv',
				'Clemson_p062_c2_grm_std_uni.csv',	'Clemson_p066_c1_grm_std_uni.csv',	'Clemson_p068_c1_grm_std_uni.csv',
				'Clemson_p070_c1_grm_std_uni.csv',	'Clemson_p072_c1_grm_std_uni.csv',	'Clemson_p077_c2_grm_std_uni.csv',
				'Clemson_p079_c2_grm_std_uni.csv',	'Clemson_p082_c2_grm_std_uni.csv',	'Clemson_p085_c1_grm_std_uni.csv',
				'Clemson_p088_c2_grm_std_uni.csv',	'Clemson_p092_c1_grm_std_uni.csv',	'Clemson_p098_c1_grm_std_uni.csv',
				'Clemson_p101_c1_grm_std_uni.csv',	'Clemson_p103_c2_grm_std_uni.csv',	'Clemson_p107_c2_grm_std_uni.csv',
				'Clemson_p110_c1_grm_std_uni.csv',	'Clemson_p114_c1_grm_std_uni.csv',	'Clemson_p116_c3_grm_std_uni.csv',
				'Clemson_p118_c1_grm_std_uni.csv',	'Clemson_p120_c2_grm_std_uni.csv',	'Clemson_p123_c2_grm_std_uni.csv',
				'Clemson_p130_c1_grm_std_uni.csv',	'Clemson_p136_c1_grm_std_uni.csv',	'Clemson_p140_c1_grm_std_uni.csv',
				'Clemson_p144_c1_grm_std_uni.csv',	'Clemson_p146_c3_grm_std_uni.csv',	'Clemson_p151_c1_grm_std_uni.csv',
				'Clemson_p157_c2_grm_std_uni.csv',	'Clemson_p160_c2_grm_std_uni.csv',	'Clemson_p165_c1_grm_std_uni.csv',
				'Clemson_p169_c2_grm_std_uni.csv',	'Clemson_p172_c2_grm_std_uni.csv',	'Clemson_p174_c3_grm_std_uni.csv',
				'Clemson_p176_c2_grm_std_uni.csv',	'Clemson_p179_c1_grm_std_uni.csv',	'Clemson_p180_c3_grm_std_uni.csv',
				'Clemson_p184_c2_grm_std_uni.csv',	'Clemson_p187_c2_grm_std_uni.csv',	'Clemson_p190_c1_grm_std_uni.csv',
				'Clemson_p194_c2_grm_std_uni.csv',	'Clemson_p198_c1_grm_std_uni.csv',	'Clemson_p204_c2_grm_std_uni.csv',
				'Clemson_p207_c1_grm_std_uni.csv',	'Clemson_p209_c1_grm_std_uni.csv',	'Clemson_p217_c1_grm_std_uni.csv',
				'Clemson_p218_c3_grm_std_uni.csv',	'Clemson_p220_c2_grm_std_uni.csv',	'Clemson_p229_c1_grm_std_uni.csv',
				'Clemson_p231_c2_grm_std_uni.csv',	'Clemson_p235_c1_grm_std_uni.csv',	'Clemson_p241_c1_grm_std_uni.csv',
				'Clemson_p244_c2_grm_std_uni.csv',	'Clemson_p248_c1_grm_std_uni.csv',	'Clemson_p253_c2_grm_std_uni.csv',
				'Clemson_p260_c1_grm_std_uni.csv',	'Clemson_p263_c1_grm_std_uni.csv',	'Clemson_p265_c2_grm_std_uni.csv',
				'Clemson_p268_c1_grm_std_uni.csv',	'Clemson_p270_c2_grm_std_uni.csv',	'Clemson_p271_c2_grm_std_uni.csv',
				'Clemson_p273_c1_grm_std_uni.csv',	'Clemson_p275_c2_grm_std_uni.csv',	'Clemson_p277_c3_grm_std_uni.csv',
				'Clemson_p279_c2_grm_std_uni.csv',	'Clemson_p281_c3_grm_std_uni.csv',	'Clemson_p285_c2_grm_std_uni.csv',
				'Clemson_p291_c2_grm_std_uni.csv',	'Clemson_p297_c2_grm_std_uni.csv',	'Clemson_p311_c1_grm_std_uni.csv',
				'Clemson_p315_c1_grm_std_uni.csv',	'Clemson_p322_c1_grm_std_uni.csv',	'Clemson_p326_c1_grm_std_uni.csv',
				'Clemson_p331_c1_grm_std_uni.csv',	'Clemson_p334_c1_grm_std_uni.csv',	'Clemson_p337_c2_grm_std_uni.csv',
				'Clemson_p343_c2_grm_std_uni.csv',	'Clemson_p352_c2_grm_std_uni.csv',	'Clemson_p361_c3_grm_std_uni.csv',
				'Clemson_p372_c2_grm_std_uni.csv',	'Clemson_p396_c1_grm_std_uni.csv',	'Clemson_p401_c1_grm_std_uni.csv',
				'Clemson_p410_c2_grm_std_uni.csv'
				],

			# FOLD 1 Files
			[
				'Clemson_p005_c1_grm_std_uni.csv',	'Clemson_p011_c1_grm_std_uni.csv',	'Clemson_p016_c1_grm_std_uni.csv',
				'Clemson_p020_c1_grm_std_uni.csv',	'Clemson_p024_c2_grm_std_uni.csv',	'Clemson_p027_c1_grm_std_uni.csv',
				'Clemson_p029_c1_grm_std_uni.csv',	'Clemson_p034_c2_grm_std_uni.csv',	'Clemson_p036_c2_grm_std_uni.csv',
				'Clemson_p042_c1_grm_std_uni.csv',	'Clemson_p045_c2_grm_std_uni.csv',	'Clemson_p050_c1_grm_std_uni.csv',
				'Clemson_p053_c1_grm_std_uni.csv',	'Clemson_p056_c1_grm_std_uni.csv',	'Clemson_p058_c1_grm_std_uni.csv',
				'Clemson_p061_c1_grm_std_uni.csv',	'Clemson_p064_c1_grm_std_uni.csv',	'Clemson_p066_c2_grm_std_uni.csv',
				'Clemson_p068_c2_grm_std_uni.csv',	'Clemson_p070_c2_grm_std_uni.csv',	'Clemson_p074_c1_grm_std_uni.csv',
				'Clemson_p077_c3_grm_std_uni.csv',	'Clemson_p080_c1_grm_std_uni.csv',	'Clemson_p083_c1_grm_std_uni.csv',
				'Clemson_p086_c1_grm_std_uni.csv',	'Clemson_p089_c1_grm_std_uni.csv',	'Clemson_p093_c1_grm_std_uni.csv',
				'Clemson_p098_c2_grm_std_uni.csv',	'Clemson_p101_c2_grm_std_uni.csv',	'Clemson_p104_c1_grm_std_uni.csv',
				'Clemson_p108_c1_grm_std_uni.csv',	'Clemson_p111_c1_grm_std_uni.csv',	'Clemson_p114_c2_grm_std_uni.csv',
				'Clemson_p117_c1_grm_std_uni.csv',	'Clemson_p118_c2_grm_std_uni.csv',	'Clemson_p121_c1_grm_std_uni.csv',
				'Clemson_p125_c1_grm_std_uni.csv',	'Clemson_p131_c1_grm_std_uni.csv',	'Clemson_p137_c1_grm_std_uni.csv',
				'Clemson_p142_c1_grm_std_uni.csv',	'Clemson_p144_c2_grm_std_uni.csv',	'Clemson_p148_c1_grm_std_uni.csv',
				'Clemson_p151_c2_grm_std_uni.csv',	'Clemson_p158_c1_grm_std_uni.csv',	'Clemson_p161_c1_grm_std_uni.csv',
				'Clemson_p165_c2_grm_std_uni.csv',	'Clemson_p170_c1_grm_std_uni.csv',	'Clemson_p173_c1_grm_std_uni.csv',
				'Clemson_p175_c1_grm_std_uni.csv',	'Clemson_p177_c1_grm_std_uni.csv',	'Clemson_p179_c2_grm_std_uni.csv',
				'Clemson_p181_c1_grm_std_uni.csv',	'Clemson_p185_c1_grm_std_uni.csv',	'Clemson_p187_c3_grm_std_uni.csv',
				'Clemson_p190_c2_grm_std_uni.csv',	'Clemson_p194_c3_grm_std_uni.csv',	'Clemson_p199_c1_grm_std_uni.csv',
				'Clemson_p204_c3_grm_std_uni.csv',	'Clemson_p207_c2_grm_std_uni.csv',	'Clemson_p209_c2_grm_std_uni.csv',
				'Clemson_p217_c2_grm_std_uni.csv',	'Clemson_p218_c4_grm_std_uni.csv',	'Clemson_p221_c1_grm_std_uni.csv',
				'Clemson_p229_c2_grm_std_uni.csv',	'Clemson_p232_c1_grm_std_uni.csv',	'Clemson_p236_c1_grm_std_uni.csv',
				'Clemson_p241_c2_grm_std_uni.csv',	'Clemson_p245_c1_grm_std_uni.csv',	'Clemson_p251_c2_grm_std_uni.csv',
				'Clemson_p256_c1_grm_std_uni.csv',	'Clemson_p260_c2_grm_std_uni.csv',	'Clemson_p263_c2_grm_std_uni.csv',
				'Clemson_p266_c1_grm_std_uni.csv',	'Clemson_p268_c2_grm_std_uni.csv',	'Clemson_p270_c3_grm_std_uni.csv',
				'Clemson_p271_c3_grm_std_uni.csv',	'Clemson_p273_c2_grm_std_uni.csv',	'Clemson_p276_c1_grm_std_uni.csv',
				'Clemson_p278_c1_grm_std_uni.csv',	'Clemson_p279_c3_grm_std_uni.csv',	'Clemson_p282_c1_grm_std_uni.csv',
				'Clemson_p289_c2_grm_std_uni.csv',	'Clemson_p292_c1_grm_std_uni.csv',	'Clemson_p298_c1_grm_std_uni.csv',
				'Clemson_p311_c2_grm_std_uni.csv',	'Clemson_p315_c2_grm_std_uni.csv',	'Clemson_p322_c2_grm_std_uni.csv',
				'Clemson_p326_c2_grm_std_uni.csv',	'Clemson_p331_c2_grm_std_uni.csv',	'Clemson_p334_c2_grm_std_uni.csv',
				'Clemson_p338_c1_grm_std_uni.csv',	'Clemson_p343_c3_grm_std_uni.csv',	'Clemson_p352_c3_grm_std_uni.csv',
				'Clemson_p368_c1_grm_std_uni.csv',	'Clemson_p377_c1_grm_std_uni.csv',	'Clemson_p396_c2_grm_std_uni.csv',
				'Clemson_p401_c2_grm_std_uni.csv',	'Clemson_p411_c1_grm_std_uni.csv'
			],

			# FOLD 2 Files
			[
				'Clemson_p005_c2_grm_std_uni.csv', 'Clemson_p011_c2_grm_std_uni.csv', 'Clemson_p016_c2_grm_std_uni.csv', 'Clemson_p021_c2_grm_std_uni.csv',
				'Clemson_p025_c1_grm_std_uni.csv', 'Clemson_p027_c2_grm_std_uni.csv', 'Clemson_p030_c1_grm_std_uni.csv', 'Clemson_p034_c3_grm_std_uni.csv',
				'Clemson_p037_c1_grm_std_uni.csv', 'Clemson_p042_c2_grm_std_uni.csv', 'Clemson_p045_c3_grm_std_uni.csv', 'Clemson_p051_c1_grm_std_uni.csv',
				'Clemson_p054_c1_grm_std_uni.csv', 'Clemson_p056_c2_grm_std_uni.csv', 'Clemson_p059_c1_grm_std_uni.csv', 'Clemson_p061_c2_grm_std_uni.csv',
				'Clemson_p064_c2_grm_std_uni.csv', 'Clemson_p066_c3_grm_std_uni.csv', 'Clemson_p069_c1_grm_std_uni.csv', 'Clemson_p070_c3_grm_std_uni.csv',
				'Clemson_p074_c2_grm_std_uni.csv', 'Clemson_p078_c1_grm_std_uni.csv', 'Clemson_p080_c2_grm_std_uni.csv', 'Clemson_p083_c2_grm_std_uni.csv',
				'Clemson_p087_c1_grm_std_uni.csv', 'Clemson_p090_c1_grm_std_uni.csv', 'Clemson_p093_c2_grm_std_uni.csv', 'Clemson_p099_c1_grm_std_uni.csv',
				'Clemson_p102_c1_grm_std_uni.csv', 'Clemson_p105_c1_grm_std_uni.csv', 'Clemson_p108_c2_grm_std_uni.csv', 'Clemson_p111_c2_grm_std_uni.csv',
				'Clemson_p115_c1_grm_std_uni.csv', 'Clemson_p117_c2_grm_std_uni.csv', 'Clemson_p119_c1_grm_std_uni.csv', 'Clemson_p121_c2_grm_std_uni.csv',
				'Clemson_p125_c2_grm_std_uni.csv', 'Clemson_p132_c1_grm_std_uni.csv', 'Clemson_p137_c2_grm_std_uni.csv', 'Clemson_p142_c2_grm_std_uni.csv',
				'Clemson_p145_c1_grm_std_uni.csv', 'Clemson_p148_c2_grm_std_uni.csv', 'Clemson_p153_c1_grm_std_uni.csv', 'Clemson_p158_c2_grm_std_uni.csv',
				'Clemson_p161_c2_grm_std_uni.csv', 'Clemson_p166_c1_grm_std_uni.csv', 'Clemson_p171_c1_grm_std_uni.csv', 'Clemson_p173_c2_grm_std_uni.csv',
				'Clemson_p175_c2_grm_std_uni.csv', 'Clemson_p177_c2_grm_std_uni.csv', 'Clemson_p179_c3_grm_std_uni.csv', 'Clemson_p182_c1_grm_std_uni.csv',
				'Clemson_p186_c1_grm_std_uni.csv', 'Clemson_p188_c1_grm_std_uni.csv', 'Clemson_p192_c1_grm_std_uni.csv', 'Clemson_p195_c1_grm_std_uni.csv',
				'Clemson_p201_c1_grm_std_uni.csv', 'Clemson_p205_c1_grm_std_uni.csv', 'Clemson_p207_c3_grm_std_uni.csv', 'Clemson_p215_c1_grm_std_uni.csv',
				'Clemson_p217_c3_grm_std_uni.csv', 'Clemson_p219_c1_grm_std_uni.csv', 'Clemson_p224_c1_grm_std_uni.csv', 'Clemson_p230_c1_grm_std_uni.csv',
				'Clemson_p233_c1_grm_std_uni.csv', 'Clemson_p236_c2_grm_std_uni.csv', 'Clemson_p242_c1_grm_std_uni.csv', 'Clemson_p246_c1_grm_std_uni.csv',
				'Clemson_p252_c1_grm_std_uni.csv', 'Clemson_p257_c1_grm_std_uni.csv', 'Clemson_p262_c1_grm_std_uni.csv', 'Clemson_p264_c1_grm_std_uni.csv',
				'Clemson_p266_c2_grm_std_uni.csv', 'Clemson_p269_c1_grm_std_uni.csv', 'Clemson_p270_c4_grm_std_uni.csv', 'Clemson_p272_c1_grm_std_uni.csv',
				'Clemson_p274_c1_grm_std_uni.csv', 'Clemson_p276_c2_grm_std_uni.csv', 'Clemson_p278_c2_grm_std_uni.csv', 'Clemson_p280_c1_grm_std_uni.csv',
				'Clemson_p283_c1_grm_std_uni.csv', 'Clemson_p290_c1_grm_std_uni.csv', 'Clemson_p293_c1_grm_std_uni.csv', 'Clemson_p298_c2_grm_std_uni.csv',
				'Clemson_p311_c3_grm_std_uni.csv', 'Clemson_p318_c1_grm_std_uni.csv', 'Clemson_p322_c3_grm_std_uni.csv', 'Clemson_p329_c1_grm_std_uni.csv',
				'Clemson_p332_c1_grm_std_uni.csv', 'Clemson_p336_c1_grm_std_uni.csv', 'Clemson_p338_c2_grm_std_uni.csv', 'Clemson_p343_c4_grm_std_uni.csv',
				'Clemson_p353_c1_grm_std_uni.csv', 'Clemson_p368_c2_grm_std_uni.csv', 'Clemson_p384_c2_grm_std_uni.csv', 'Clemson_p396_c3_grm_std_uni.csv',
				'Clemson_p406_c1_grm_std_uni.csv', 'Clemson_p411_c2_grm_std_uni.csv'
			],

			# FOLD 3 Files
			[
				'Clemson_p006_c1_grm_std_uni.csv', 'Clemson_p012_c2_grm_std_uni.csv', 'Clemson_p017_c2_grm_std_uni.csv', 'Clemson_p022_c1_grm_std_uni.csv',
				'Clemson_p026_c1_grm_std_uni.csv', 'Clemson_p027_c3_grm_std_uni.csv', 'Clemson_p031_c1_grm_std_uni.csv', 'Clemson_p035_c1_grm_std_uni.csv',
				'Clemson_p037_c2_grm_std_uni.csv', 'Clemson_p043_c1_grm_std_uni.csv', 'Clemson_p046_c1_grm_std_uni.csv', 'Clemson_p051_c2_grm_std_uni.csv',
				'Clemson_p054_c2_grm_std_uni.csv', 'Clemson_p057_c1_grm_std_uni.csv', 'Clemson_p060_c1_grm_std_uni.csv', 'Clemson_p061_c3_grm_std_uni.csv',
				'Clemson_p065_c1_grm_std_uni.csv', 'Clemson_p067_c1_grm_std_uni.csv', 'Clemson_p069_c3_grm_std_uni.csv', 'Clemson_p071_c1_grm_std_uni.csv',
				'Clemson_p075_c1_grm_std_uni.csv', 'Clemson_p078_c2_grm_std_uni.csv', 'Clemson_p081_c1_grm_std_uni.csv', 'Clemson_p084_c1_grm_std_uni.csv',
				'Clemson_p087_c2_grm_std_uni.csv', 'Clemson_p090_c2_grm_std_uni.csv', 'Clemson_p095_c1_grm_std_uni.csv', 'Clemson_p099_c2_grm_std_uni.csv',
				'Clemson_p102_c2_grm_std_uni.csv', 'Clemson_p106_c1_grm_std_uni.csv', 'Clemson_p109_c1_grm_std_uni.csv', 'Clemson_p113_c1_grm_std_uni.csv',
				'Clemson_p115_c2_grm_std_uni.csv', 'Clemson_p117_c3_grm_std_uni.csv', 'Clemson_p119_c2_grm_std_uni.csv', 'Clemson_p122_c1_grm_std_uni.csv',
				'Clemson_p129_c1_grm_std_uni.csv', 'Clemson_p132_c2_grm_std_uni.csv', 'Clemson_p138_c1_grm_std_uni.csv', 'Clemson_p143_c1_grm_std_uni.csv',
				'Clemson_p146_c1_grm_std_uni.csv', 'Clemson_p150_c1_grm_std_uni.csv', 'Clemson_p154_c1_grm_std_uni.csv', 'Clemson_p159_c1_grm_std_uni.csv',
				'Clemson_p162_c1_grm_std_uni.csv', 'Clemson_p166_c2_grm_std_uni.csv', 'Clemson_p171_c2_grm_std_uni.csv', 'Clemson_p174_c1_grm_std_uni.csv',
				'Clemson_p175_c3_grm_std_uni.csv', 'Clemson_p178_c1_grm_std_uni.csv', 'Clemson_p180_c1_grm_std_uni.csv', 'Clemson_p182_c2_grm_std_uni.csv',
				'Clemson_p186_c2_grm_std_uni.csv', 'Clemson_p188_c2_grm_std_uni.csv', 'Clemson_p192_c2_grm_std_uni.csv', 'Clemson_p195_c2_grm_std_uni.csv',
				'Clemson_p202_c1_grm_std_uni.csv', 'Clemson_p205_c2_grm_std_uni.csv', 'Clemson_p208_c1_grm_std_uni.csv', 'Clemson_p215_c2_grm_std_uni.csv',
				'Clemson_p218_c1_grm_std_uni.csv', 'Clemson_p219_c2_grm_std_uni.csv', 'Clemson_p226_c1_grm_std_uni.csv', 'Clemson_p230_c2_grm_std_uni.csv',
				'Clemson_p234_c1_grm_std_uni.csv', 'Clemson_p237_c1_grm_std_uni.csv', 'Clemson_p242_c2_grm_std_uni.csv', 'Clemson_p247_c1_grm_std_uni.csv',
				'Clemson_p252_c2_grm_std_uni.csv', 'Clemson_p257_c2_grm_std_uni.csv', 'Clemson_p262_c2_grm_std_uni.csv', 'Clemson_p264_c2_grm_std_uni.csv',
				'Clemson_p267_c1_grm_std_uni.csv', 'Clemson_p269_c2_grm_std_uni.csv', 'Clemson_p270_c5_grm_std_uni.csv', 'Clemson_p272_c2_grm_std_uni.csv',
				'Clemson_p274_c2_grm_std_uni.csv', 'Clemson_p277_c1_grm_std_uni.csv', 'Clemson_p278_c3_grm_std_uni.csv', 'Clemson_p280_c2_grm_std_uni.csv',
				'Clemson_p284_c1_grm_std_uni.csv', 'Clemson_p290_c2_grm_std_uni.csv', 'Clemson_p293_c2_grm_std_uni.csv', 'Clemson_p309_c1_grm_std_uni.csv',
				'Clemson_p312_c1_grm_std_uni.csv', 'Clemson_p320_c1_grm_std_uni.csv', 'Clemson_p324_c1_grm_std_uni.csv', 'Clemson_p329_c2_grm_std_uni.csv',
				'Clemson_p332_c2_grm_std_uni.csv', 'Clemson_p336_c2_grm_std_uni.csv', 'Clemson_p341_c1_grm_std_uni.csv', 'Clemson_p347_c1_grm_std_uni.csv',
				'Clemson_p353_c2_grm_std_uni.csv', 'Clemson_p368_c3_grm_std_uni.csv', 'Clemson_p392_c1_grm_std_uni.csv', 'Clemson_p397_c1_grm_std_uni.csv',
				'Clemson_p406_c2_grm_std_uni.csv', 'Clemson_p413_c1_grm_std_uni.csv'
			],

			# FOLD 4 Files
			[
				'Clemson_p006_c2_grm_std_uni.csv', 'Clemson_p013_c1_grm_std_uni.csv', 'Clemson_p019_c1_grm_std_uni.csv', 'Clemson_p023_c1_grm_std_uni.csv',
				'Clemson_p026_c2_grm_std_uni.csv', 'Clemson_p028_c1_grm_std_uni.csv', 'Clemson_p033_c1_grm_std_uni.csv', 'Clemson_p035_c2_grm_std_uni.csv',
				'Clemson_p038_c1_grm_std_uni.csv', 'Clemson_p044_c1_grm_std_uni.csv', 'Clemson_p047_c2_grm_std_uni.csv', 'Clemson_p052_c1_grm_std_uni.csv',
				'Clemson_p055_c1_grm_std_uni.csv', 'Clemson_p057_c2_grm_std_uni.csv', 'Clemson_p060_c2_grm_std_uni.csv', 'Clemson_p062_c1_grm_std_uni.csv',
				'Clemson_p065_c2_grm_std_uni.csv', 'Clemson_p067_c2_grm_std_uni.csv', 'Clemson_p069_c4_grm_std_uni.csv', 'Clemson_p071_c2_grm_std_uni.csv',
				'Clemson_p077_c1_grm_std_uni.csv', 'Clemson_p079_c1_grm_std_uni.csv', 'Clemson_p082_c1_grm_std_uni.csv', 'Clemson_p084_c2_grm_std_uni.csv',
				'Clemson_p088_c1_grm_std_uni.csv', 'Clemson_p091_c1_grm_std_uni.csv', 'Clemson_p096_c1_grm_std_uni.csv', 'Clemson_p100_c1_grm_std_uni.csv',
				'Clemson_p103_c1_grm_std_uni.csv', 'Clemson_p107_c1_grm_std_uni.csv', 'Clemson_p109_c2_grm_std_uni.csv', 'Clemson_p113_c2_grm_std_uni.csv',
				'Clemson_p116_c2_grm_std_uni.csv', 'Clemson_p117_c4_grm_std_uni.csv', 'Clemson_p120_c1_grm_std_uni.csv', 'Clemson_p122_c2_grm_std_uni.csv',
				'Clemson_p129_c2_grm_std_uni.csv', 'Clemson_p133_c1_grm_std_uni.csv', 'Clemson_p139_c1_grm_std_uni.csv', 'Clemson_p143_c2_grm_std_uni.csv',
				'Clemson_p146_c2_grm_std_uni.csv', 'Clemson_p150_c2_grm_std_uni.csv', 'Clemson_p157_c1_grm_std_uni.csv', 'Clemson_p160_c1_grm_std_uni.csv',
				'Clemson_p164_c1_grm_std_uni.csv', 'Clemson_p169_c1_grm_std_uni.csv', 'Clemson_p172_c1_grm_std_uni.csv', 'Clemson_p174_c2_grm_std_uni.csv', 
				'Clemson_p176_c1_grm_std_uni.csv', 'Clemson_p178_c2_grm_std_uni.csv', 'Clemson_p180_c2_grm_std_uni.csv', 'Clemson_p184_c1_grm_std_uni.csv', 
				'Clemson_p187_c1_grm_std_uni.csv', 'Clemson_p189_c1_grm_std_uni.csv', 'Clemson_p194_c1_grm_std_uni.csv', 'Clemson_p195_c3_grm_std_uni.csv', 
				'Clemson_p204_c1_grm_std_uni.csv', 'Clemson_p206_c1_grm_std_uni.csv', 'Clemson_p208_c2_grm_std_uni.csv', 'Clemson_p215_c3_grm_std_uni.csv',
				'Clemson_p218_c2_grm_std_uni.csv', 'Clemson_p220_c1_grm_std_uni.csv', 'Clemson_p226_c2_grm_std_uni.csv', 'Clemson_p231_c1_grm_std_uni.csv',
				'Clemson_p234_c2_grm_std_uni.csv', 'Clemson_p237_c2_grm_std_uni.csv', 'Clemson_p244_c1_grm_std_uni.csv', 'Clemson_p247_c2_grm_std_uni.csv',
				'Clemson_p253_c1_grm_std_uni.csv', 'Clemson_p259_c1_grm_std_uni.csv', 'Clemson_p262_c3_grm_std_uni.csv', 'Clemson_p265_c1_grm_std_uni.csv',
				'Clemson_p267_c2_grm_std_uni.csv', 'Clemson_p270_c1_grm_std_uni.csv', 'Clemson_p271_c1_grm_std_uni.csv', 'Clemson_p272_c3_grm_std_uni.csv',
				'Clemson_p275_c1_grm_std_uni.csv', 'Clemson_p277_c2_grm_std_uni.csv', 'Clemson_p279_c1_grm_std_uni.csv', 'Clemson_p281_c1_grm_std_uni.csv',
				'Clemson_p285_c1_grm_std_uni.csv', 'Clemson_p291_c1_grm_std_uni.csv', 'Clemson_p297_c1_grm_std_uni.csv', 'Clemson_p309_c2_grm_std_uni.csv',
				'Clemson_p312_c2_grm_std_uni.csv', 'Clemson_p320_c2_grm_std_uni.csv', 'Clemson_p324_c2_grm_std_uni.csv', 'Clemson_p329_c3_grm_std_uni.csv',
				'Clemson_p332_c3_grm_std_uni.csv', 'Clemson_p337_c1_grm_std_uni.csv', 'Clemson_p343_c1_grm_std_uni.csv', 'Clemson_p352_c1_grm_std_uni.csv',
				'Clemson_p353_c3_grm_std_uni.csv', 'Clemson_p372_c1_grm_std_uni.csv', 'Clemson_p392_c2_grm_std_uni.csv', 'Clemson_p397_c2_grm_std_uni.csv',
				'Clemson_p410_c1_grm_std_uni.csv'
			]
		]
		
	# end of switch DatasetFlag
		
		
	return ResultsFilenames
	
# end of def GetResultFilenames()

