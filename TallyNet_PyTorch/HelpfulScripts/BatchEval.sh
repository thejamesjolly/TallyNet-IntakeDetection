# ####
# Script to train many TallyNet Models with various architectures
####

DetMethod=1

# EvalMethod=2
# WinTol=08

# EvalMethod=1
# WinTol=99

EvalMethod=2
WinTol=12

archSelection=333 # Architecture count = ConvLayer - LSTM Layer - Dense Layer 

# for WindowSizeSec in {5..25..5}
# for WindowSizeSec in {5..10..5}
for WindowSizeSec in {5..25..5}
do
	
	for Instance in {0..5}
	do
		
		echo "Scheduling DomCleanClem_A1"$archSelection"_WS"$WindowSizeSec"_s05_"$Instance" Five Fold with Eval="$EvalMethod" and WinTol="$WinTol""
		sbatch ./batch_script_TallyNet_Eval.sh "DomCleanClem_A1"$archSelection"_WS"$WindowSizeSec"_s05_"$Instance"" $WindowSizeSec 5 $DetMethod $EvalMethod $WinTol


		echo "Scheduling DomOHO_A1"$archSelection"_WS"$WindowSizeSec"_s03_"$Instance" Five Fold with Eval="$EvalMethod" and WinTol="$WinTol""
		sbatch ./batch_script_TallyNet_Eval.sh "DomOHOv2_A1"$archSelection"_WS"$WindowSizeSec"_s03_"$Instance"" $WindowSizeSec 6 $DetMethod $EvalMethod $WinTol
		
	done
	
done




echo -e "\n\nFINISHED SCHEDULING ALL EVAL JOBS\n"









<< ARCHITECTURE_SEARCH

for Filters in {3..3}
do
	for Lstm in {3..3}
	do
		for Dense in {3..3}
		do
			archSelection=$Filters$Lstm$Dense
			### EXAMPLE ### 
			# ./batch_script_TallyNet_Eval.sh "ModelFolderName" "Database_Flag" "Det_Method" "Eval_Method_Flag" "WinTol"
			# echo "\nScheduling Eval of A1"$archSelection"_0 Five Fold"
			# sbatch ./batch_script_TallyNet_Eval.sh "Clem_A1"$archSelection"_0" 3 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_Eval.sh "OHO_A1"$archSelection"_0" 1 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_Eval.sh "THO_A1"$archSelection"_0" 2 $DetMethod $EvalMethod $WinTol

			echo "\nScheduling Eval Clem of A1"$archSelection"_s05_0to3 Five Fold"
			sbatch ./batch_script_TallyNet_Eval.sh "Clem_A1"$archSelection"_s05_0" 4 $DetMethod $EvalMethod $WinTol
			sbatch ./batch_script_TallyNet_Eval.sh "Clem_A1"$archSelection"_s05_1" 4 $DetMethod $EvalMethod $WinTol
			sbatch ./batch_script_TallyNet_Eval.sh "Clem_A1"$archSelection"_s05_2" 4 $DetMethod $EvalMethod $WinTol
			sbatch ./batch_script_TallyNet_Eval.sh "Clem_A1"$archSelection"_s05_3" 4 $DetMethod $EvalMethod $WinTol


# 			echo "\nScheduling Eval OHO of A1"$archSelection"_2 Five Fold"
# 			sbatch ./batch_script_TallyNet_Eval.sh "OHO_A1"$archSelection"_s03_3" 1 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_Eval.sh "OHO_A1"$archSelection"_s03_4" 1 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_Eval.sh "OHO_A1"$archSelection"_s03_5" 1 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_Eval.sh "OHO_A1"$archSelection"_s03_6" 1 $DetMethod $EvalMethod $WinTol


# 			echo "\nScheduling Eval THO of A1"$archSelection"_1 Five Fold"
# 			sbatch ./batch_script_TallyNet_Eval.sh "THO_A1"$archSelection"_s03_3" 2 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_Eval.sh "THO_A1"$archSelection"_s03_4" 2 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_Eval.sh "THO_A1"$archSelection"_s03_5" 2 $DetMethod $EvalMethod $WinTol
# 			sbatch ./batch_script_TallyNet_Eval.sh "THO_A1"$archSelection"_s03_6" 2 $DetMethod $EvalMethod $WinTol


			# sbatch ./batch_script_TallyNet_Eval.sh "Clem_A1"$archSelection"_s05_2" 3 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_Eval.sh "OHO_A1"$archSelection"_s03_2" 1 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_Eval.sh "THO_A1"$archSelection"_s05_2" 2 $DetMethod $EvalMethod $WinTol


			# sbatch ./batch_script_TallyNet_Eval.sh "Clem_A1"$archSelection"_s05_1" 3 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_Eval.sh "OHO_A1"$archSelection"_s03_1" 1 $DetMethod $EvalMethod $WinTol
			# sbatch ./batch_script_TallyNet_Eval.sh "THO_A1"$archSelection"_s03_1" 2 $DetMethod $EvalMethod $WinTol

		done
	done
done

ARCHITECTURE_SEARCH

