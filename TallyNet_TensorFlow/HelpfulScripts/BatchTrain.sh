# ####
# Script to train many TallyNet Models with various architectures
####


archSelection=333 # Architecture count = ConvLayer - LSTM Layer - Dense Layer 

for WindowSizeSec in {5..10..5}
# for WindowSizeSec in {15..15}
do

    for Instance in {0..5}
	do
		# echo "./batch_script_TallyNet_Train.sh \"Models/Clem_A1"$archSelection"_s05_0\" 3 0 1$archSelection"

		# echo "Scheduling Clem_A1"$archSelection"_WS"$WindowSizeSec"_s05_"$Instance" Five Fold"
		# sbatch ./batch_script_TallyNet_Train_DomCleanClem.sh "Models/DomCleanClem_A1"$archSelection"_WS"$WindowSizeSec"_s05_"$Instance"" 0 1$archSelection $WindowSizeSec
		# sbatch ./batch_script_TallyNet_Train_DomCleanClem.sh "Models/DomCleanClem_A1"$archSelection"_WS"$WindowSizeSec"_s05_"$Instance"" 1 1$archSelection $WindowSizeSec
		# sbatch ./batch_script_TallyNet_Train_DomCleanClem.sh "Models/DomCleanClem_A1"$archSelection"_WS"$WindowSizeSec"_s05_"$Instance"" 2 1$archSelection $WindowSizeSec
		# sbatch ./batch_script_TallyNet_Train_DomCleanClem.sh "Models/DomCleanClem_A1"$archSelection"_WS"$WindowSizeSec"_s05_"$Instance"" 3 1$archSelection $WindowSizeSec
		# sbatch ./batch_script_TallyNet_Train_DomCleanClem.sh "Models/DomCleanClem_A1"$archSelection"_WS"$WindowSizeSec"_s05_"$Instance"" 4 1$archSelection $WindowSizeSec


		echo "Scheduling OHO_A1"$archSelection"_WS"$WindowSizeSec"_s03_"$Instance" Five Fold"
		sbatch ./batch_script_TallyNet_Train_DomOHO.sh "Models/DomOHOv2_A1"$archSelection"_WS"$WindowSizeSec"_s03_"$Instance"" 0 1$archSelection $WindowSizeSec
		sbatch ./batch_script_TallyNet_Train_DomOHO.sh "Models/DomOHOv2_A1"$archSelection"_WS"$WindowSizeSec"_s03_"$Instance"" 1 1$archSelection $WindowSizeSec
		sbatch ./batch_script_TallyNet_Train_DomOHO.sh "Models/DomOHOv2_A1"$archSelection"_WS"$WindowSizeSec"_s03_"$Instance"" 2 1$archSelection $WindowSizeSec
		sbatch ./batch_script_TallyNet_Train_DomOHO.sh "Models/DomOHOv2_A1"$archSelection"_WS"$WindowSizeSec"_s03_"$Instance"" 3 1$archSelection $WindowSizeSec
		sbatch ./batch_script_TallyNet_Train_DomOHO.sh "Models/DomOHOv2_A1"$archSelection"_WS"$WindowSizeSec"_s03_"$Instance"" 4 1$archSelection $WindowSizeSec


	done
	
	echo "Finished Scheduling jobs with window size of "$WindowSizeSec"."
done






<<YET_TO_RUN


##### USED FOR ARCHITECTURE SEARCHING

for Filters in {3..3}
do
	for Lstm in {3..3}
	do
		for Dense in {3..3}
		do
			archSelection=$Filters$Lstm$Dense
			# echo "./batch_script_TallyNet_Train.sh \"Models/Clem_A1"$archSelection"_s05_0\" 3 0 1$archSelection"

			echo "Scheduling Clem_A1"$archSelection"_s05_8 Five Fold mem=20GB"
			sbatch ./batch_script_TallyNet_Train.sh "Models/CleanClem_A1"$archSelection"_s05_3" 4 0 1$archSelection
			sbatch ./batch_script_TallyNet_Train.sh "Models/CleanClem_A1"$archSelection"_s05_3" 4 1 1$archSelection
			sbatch ./batch_script_TallyNet_Train.sh "Models/CleanClem_A1"$archSelection"_s05_3" 4 2 1$archSelection
			sbatch ./batch_script_TallyNet_Train.sh "Models/CleanClem_A1"$archSelection"_s05_3" 4 3 1$archSelection
			sbatch ./batch_script_TallyNet_Train.sh "Models/CleanClem_A1"$archSelection"_s05_3" 4 4 1$archSelection

			# echo "Scheduling OHO_A1"$archSelection"_s03_8 Five Fold"
			# sbatch ./batch_script_TallyNet_Train.sh "Models/OHO_A1"$archSelection"_s03_8" 1 0 1$archSelection
			# sbatch ./batch_script_TallyNet_Train.sh "Models/OHO_A1"$archSelection"_s03_8" 1 1 1$archSelection
			# sbatch ./batch_script_TallyNet_Train.sh "Models/OHO_A1"$archSelection"_s03_8" 1 2 1$archSelection
			# sbatch ./batch_script_TallyNet_Train.sh "Models/OHO_A1"$archSelection"_s03_8" 1 3 1$archSelection
			# sbatch ./batch_script_TallyNet_Train.sh "Models/OHO_A1"$archSelection"_s03_8" 1 4 1$archSelection

			# echo "Scheduling THO_A1"$archSelection"_s03_8 Five Fold"
			# sbatch ./batch_script_TallyNet_Train.sh "Models/THO_A1"$archSelection"_s03_8" 2 0 1$archSelection
			# sbatch ./batch_script_TallyNet_Train.sh "Models/THO_A1"$archSelection"_s03_8" 2 1 1$archSelection
			# sbatch ./batch_script_TallyNet_Train.sh "Models/THO_A1"$archSelection"_s03_8" 2 2 1$archSelection
			# sbatch ./batch_script_TallyNet_Train.sh "Models/THO_A1"$archSelection"_s03_8" 2 3 1$archSelection
			# sbatch ./batch_script_TallyNet_Train.sh "Models/THO_A1"$archSelection"_s03_8" 2 4 1$archSelection
		done
	done
done



YET_TO_RUN

