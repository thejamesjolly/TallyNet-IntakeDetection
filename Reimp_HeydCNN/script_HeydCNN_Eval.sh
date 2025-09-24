#!/bin/bash

#SBATCH --job-name Eval-HeydCNN
#SBATCH --output=./joboutput/HeydEval-slurm-%j.out
#SBATCH --error=./joboutput/HeydEval-slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 7gb
#SBATCH --time 4:00:00
### REMOVED #### SBATCH --gpus-per-node 1

cd ~/TallyNet_Repo_v2/Reimp_HeydCNN

module load anaconda3/2023.09-0 
module load cuda


source activate tf_RCF_env


# python Eval_HeydCNN.py "Folder_Name"/models/fold1 "FoldNum" "FoldTotal" "FoldSplitFlag" "Stride_s" "DatabaseFlag" "WinTol" >> "Results_Filename"

# python Eval_HeydCNN.py Example_HeydCNN_Clemson/models/fold3 3 5 1 2 3 12 >> Results/ExClem_Results_"$RunID".txt

EvalSwitch=$1
PartialSwitch=$2

if [[ $EvalSwitch -eq 1 ]]; then
	echo "Running inside Clemson-DND Segment..."
	RunID="vPaper"

	if [[ $PartialSwitch -eq 1 ]]; then
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_0/models/fold1 3 1 0.65 > Results/DndClemson_0_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_0/models/fold2 3 2 0.65 >> Results/DndClemson_0_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_0/models/fold3 3 3 0.65 >> Results/DndClemson_0_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_0/models/fold4 3 4 0.65 >> Results/DndClemson_0_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_0/models/fold0 3 0 0.65 >> Results/DndClemson_0_"$RunID".txt
	elif [[ $PartialSwitch -eq 2 ]]; then
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_1/models/fold1 3 1 0.65 > Results/DndClemson_1_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_1/models/fold2 3 2 0.65 >> Results/DndClemson_1_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_1/models/fold3 3 3 0.65 >> Results/DndClemson_1_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_1/models/fold4 3 4 0.65 >> Results/DndClemson_1_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_1/models/fold0 3 0 0.65 >> Results/DndClemson_1_"$RunID".txt
	elif [[ $PartialSwitch -eq 3 ]]; then
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_2/models/fold1 3 1 0.65 > Results/DndClemson_2_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_2/models/fold2 3 2 0.65 >> Results/DndClemson_2_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_2/models/fold3 3 3 0.65 >> Results/DndClemson_2_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_2/models/fold4 3 4 0.65 >> Results/DndClemson_2_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_2/models/fold0 3 0 0.65 >> Results/DndClemson_2_"$RunID".txt
	elif [[ $PartialSwitch -eq 4 ]]; then
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_3/models/fold1 3 1 0.65 > Results/DndClemson_3_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_3/models/fold2 3 2 0.65 >> Results/DndClemson_3_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_3/models/fold3 3 3 0.65 >> Results/DndClemson_3_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_3/models/fold4 3 4 0.65 >> Results/DndClemson_3_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_3/models/fold0 3 0 0.65 >> Results/DndClemson_3_"$RunID".txt
	elif [[ $PartialSwitch -eq 5 ]]; then
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_4/models/fold1 3 1 0.65 > Results/DndClemson_4_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_4/models/fold2 3 2 0.65 >> Results/DndClemson_4_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_4/models/fold3 3 3 0.65 >> Results/DndClemson_4_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_4/models/fold4 3 4 0.65 >> Results/DndClemson_4_"$RunID".txt
		python Eval_HeydCNN.py Models/ClemDND_HeydCNN_4/models/fold0 3 0 0.65 >> Results/DndClemson_4_"$RunID".txt
	fi

elif [[ $EvalSwitch -eq 2 ]]; then
	echo "Running inside OHO_DND Segment..."
	RunID="vPaper"
	if [[ $PartialSwitch -eq 1 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_OH_0/models/fold1 1 1 0.25 > Results/DndOHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_0/models/fold2 1 2 0.20 >> Results/DndOHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_0/models/fold3 1 3 0.30 >> Results/DndOHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_0/models/fold4 1 4 0.35 >> Results/DndOHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_0/models/fold0 1 0 0.30 >> Results/DndOHO_0_"$RunID".txt
	elif [[ $PartialSwitch -eq 2 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_OH_1/models/fold1 1 1 0.35 > Results/DndOHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_1/models/fold2 1 2 0.40 >> Results/DndOHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_1/models/fold3 1 3 0.25 >> Results/DndOHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_1/models/fold4 1 4 0.30 >> Results/DndOHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_1/models/fold0 1 0 0.30 >> Results/DndOHO_1_"$RunID".txt
	elif [[ $PartialSwitch -eq 3 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_OH_2/models/fold1 1 1 0.33 > Results/DndOHO_2_"$RunID".txt # Somewhere between 0.30 and 0.35
		python Eval_HeydCNN.py Models/HeydCNN_OH_2/models/fold2 1 2 0.40 >> Results/DndOHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_2/models/fold3 1 3 0.20 >> Results/DndOHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_2/models/fold4 1 4 0.25 >> Results/DndOHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_2/models/fold0 1 0 0.40 >> Results/DndOHO_2_"$RunID".txt
	elif [[ $PartialSwitch -eq 4 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_OH_3/models/fold1 1 1 0.30 > Results/DndOHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_3/models/fold2 1 2 0.40 >> Results/DndOHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_3/models/fold3 1 3 0.35 >> Results/DndOHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_3/models/fold4 1 4 0.35 >> Results/DndOHO_3_"$RunID".txt # Maybe keep 0.30
		python Eval_HeydCNN.py Models/HeydCNN_OH_3/models/fold0 1 0 0.25 >> Results/DndOHO_3_"$RunID".txt
	elif [[ $PartialSwitch -eq 5 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_OH_4/models/fold1 1 1 0.35 > Results/DndOHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_4/models/fold2 1 2 0.30 >> Results/DndOHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_4/models/fold3 1 3 0.20 >> Results/DndOHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_4/models/fold4 1 4 0.25 >> Results/DndOHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_OH_4/models/fold0 1 0 0.20 >> Results/DndOHO_4_"$RunID".txt
	fi
	
elif [[ $EvalSwitch -eq 3 ]]; then
	echo "Running inside THO Segment..."
	RunID="vPaper"
	if [[ $PartialSwitch -eq 1 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_DM_0/models/fold1 2 1 0.60 > Results/DndTHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_0/models/fold2 2 2 0.50 >> Results/DndTHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_0/models/fold3 2 3 0.55 >> Results/DndTHO_0_"$RunID".txt # Maybe keep 0.60
		python Eval_HeydCNN.py Models/HeydCNN_DM_0/models/fold4 2 4 0.55 >> Results/DndTHO_0_"$RunID".txt # Maybe keep 0.60
		python Eval_HeydCNN.py Models/HeydCNN_DM_0/models/fold0 2 0 0.55 >> Results/DndTHO_0_"$RunID".txt
	elif [[ $PartialSwitch -eq 2 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_DM_1/models/fold1 2 1 0.70 > Results/DndTHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_1/models/fold2 2 2 0.65 >> Results/DndTHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_1/models/fold3 2 3 0.70 >> Results/DndTHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_1/models/fold4 2 4 0.65 >> Results/DndTHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_1/models/fold0 2 0 0.55 >> Results/DndTHO_1_"$RunID".txt
	elif [[ $PartialSwitch -eq 3 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_DM_2/models/fold1 2 1 0.55 > Results/DndTHO_2_"$RunID".txt # Maybe keep 0.60
		python Eval_HeydCNN.py Models/HeydCNN_DM_2/models/fold2 2 2 0.65 >> Results/DndTHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_2/models/fold3 2 3 0.70 >> Results/DndTHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_2/models/fold4 2 4 0.55 >> Results/DndTHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_2/models/fold0 2 0 0.50 >> Results/DndTHO_2_"$RunID".txt
	elif [[ $PartialSwitch -eq 4 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_DM_3/models/fold1 2 1 0.60 > Results/DndTHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_3/models/fold2 2 2 0.60 >> Results/DndTHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_3/models/fold3 2 3 0.65 >> Results/DndTHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_3/models/fold4 2 4 0.50 >> Results/DndTHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_3/models/fold0 2 0 0.60 >> Results/DndTHO_3_"$RunID".txt
	elif [[ $PartialSwitch -eq 5 ]]; then
		python Eval_HeydCNN.py Models/HeydCNN_DM_4/models/fold1 2 1 0.60 > Results/DndTHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_4/models/fold2 2 2 0.65 >> Results/DndTHO_4_"$RunID".txt # Maybe keep 0.60
		python Eval_HeydCNN.py Models/HeydCNN_DM_4/models/fold3 2 3 0.65 >> Results/DndTHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_4/models/fold4 2 4 0.55 >> Results/DndTHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/HeydCNN_DM_4/models/fold0 2 0 0.60 >> Results/DndTHO_4_"$RunID".txt
	fi



elif [[ $EvalSwitch -eq 4 ]]; then

	echo "Running inside OHDom Segment..."
	RunID="vPaper"
	if [[ $PartialSwitch -eq 1 ]]; then
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_0/models/fold1 4 1 0.65 > Results/DomOHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_0/models/fold2 4 2 0.55 >> Results/DomOHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_0/models/fold3 4 3 0.60 >> Results/DomOHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_0/models/fold4 4 4 0.65 >> Results/DomOHO_0_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_0/models/fold0 4 0 0.55 >> Results/DomOHO_0_"$RunID".txt
	elif [[ $PartialSwitch -eq 2 ]]; then
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_1/models/fold1 4 1 0.55 > Results/DomOHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_1/models/fold2 4 2 0.70 >> Results/DomOHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_1/models/fold3 4 3 0.65 >> Results/DomOHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_1/models/fold4 4 4 0.60 >> Results/DomOHO_1_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_1/models/fold0 4 0 0.60 >> Results/DomOHO_1_"$RunID".txt
	elif [[ $PartialSwitch -eq 3 ]]; then
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_2/models/fold1 4 1 0.50 > Results/DomOHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_2/models/fold2 4 2 0.50 >> Results/DomOHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_2/models/fold3 4 3 0.60 >> Results/DomOHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_2/models/fold4 4 4 0.65 >> Results/DomOHO_2_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_2/models/fold0 4 0 0.55 >> Results/DomOHO_2_"$RunID".txt
	elif [[ $PartialSwitch -eq 4 ]]; then
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_3/models/fold1 4 1 0.55 > Results/DomOHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_3/models/fold2 4 2 0.50 >> Results/DomOHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_3/models/fold3 4 3 0.70 >> Results/DomOHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_3/models/fold4 4 4 0.55 >> Results/DomOHO_3_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_3/models/fold0 4 0 0.65 >> Results/DomOHO_3_"$RunID".txt
	elif [[ $PartialSwitch -eq 5 ]]; then
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_4/models/fold1 4 1 0.50 > Results/DomOHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_4/models/fold2 4 2 0.55 >> Results/DomOHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_4/models/fold3 4 3 0.55 >> Results/DomOHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_4/models/fold4 4 4 0.60 >> Results/DomOHO_4_"$RunID".txt
		python Eval_HeydCNN.py Models/DomOHO_HeydCNN_4/models/fold0 4 0 0.55 >> Results/DomOHO_4_"$RunID".txt
	fi

elif [[ $EvalSwitch -eq 5 ]]; then
	echo "Running inside Clemson-Dom Segment..."
	RunID="vPaper"

	if [[ $PartialSwitch -eq 1 ]]; then
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_0/models/fold1 5 1 0.70 > Results/DomClem_0_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_0/models/fold2 5 2 0.50 >> Results/DomClem_0_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_0/models/fold3 5 3 0.75 >> Results/DomClem_0_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_0/models/fold4 5 4 0.65 >> Results/DomClem_0_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_0/models/fold0 5 0 0.65 >> Results/DomClem_0_"$RunID".txt
	elif [[ $PartialSwitch -eq 2 ]]; then
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_1/models/fold1 5 1 0.60 > Results/DomClem_1_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_1/models/fold2 5 2 0.70 >> Results/DomClem_1_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_1/models/fold3 5 3 0.70 >> Results/DomClem_1_"$RunID".txt # Maybe keep 0.65
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_1/models/fold4 5 4 0.70 >> Results/DomClem_1_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_1/models/fold0 5 0 0.70 >> Results/DomClem_1_"$RunID".txt # Maybe keep 0.65
	elif [[ $PartialSwitch -eq 3 ]]; then
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_2/models/fold1 5 1 0.75 > Results/DomClem_2_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_2/models/fold2 5 2 0.75 >> Results/DomClem_2_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_2/models/fold3 5 3 0.75 >> Results/DomClem_2_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_2/models/fold4 5 4 0.55 >> Results/DomClem_2_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_2/models/fold0 5 0 0.65 >> Results/DomClem_2_"$RunID".txt
	elif [[ $PartialSwitch -eq 4 ]]; then
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_3/models/fold1 5 1 0.75 > Results/DomClem_3_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_3/models/fold2 5 2 0.65 >> Results/DomClem_3_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_3/models/fold3 5 3 0.65 >> Results/DomClem_3_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_3/models/fold4 5 4 0.65 >> Results/DomClem_3_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_3/models/fold0 5 0 0.75 >> Results/DomClem_3_"$RunID".txt
	elif [[ $PartialSwitch -eq 5 ]]; then
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_4/models/fold1 5 1 0.65 > Results/DomClem_4_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_4/models/fold2 5 2 0.65 >> Results/DomClem_4_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_4/models/fold3 5 3 0.65 >> Results/DomClem_4_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_4/models/fold4 5 4 0.65 >> Results/DomClem_4_"$RunID".txt
		python Eval_HeydCNN.py Models/DomClem_HeydCNN_4/models/fold0 5 0 0.65 >> Results/DomClem_4_"$RunID".txt
	fi

fi

echo "FINISHED EVALUATING."
