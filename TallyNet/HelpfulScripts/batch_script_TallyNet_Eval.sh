#!/bin/bash


#SBATCH --job-name TN-ev-Dom
#SBATCH --output=./joboutput/TN_Ev_Dom_slurm-%j.out
#SBATCH --error=./joboutput/TN_Ev_Dom_slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 10gb
#SBATCH --time 2:00:00
#### REMOVING GPU #SBA###TCH --gpus-per-node 1

cd ~/TallyNet_Experiments/TallyNet_Dom_Models

module load anaconda3/2023.09-0 
module load cuda


source activate tf_v2.2_env


Folder_Name=$1 #Folder Name DOES NOT END IN '/'
WindowSizeSec=$2
DatabaseFlag=$3 # which database to train on; 1=OneHand Oreba, 2=TwoHand Oreba, 3=ClemCafe
DetectMethod=$4 # which placement method to use for TallyNet: midpoint, tetris, midpt tetris, etc.
EvalMethod=$5 # which Evaluation method to use: 1 = PtVsPt, 2 = PtVsWindow, DurationWindow, etc.
Tol=$6 # Tolerance for the detection method if needed

STRIDE_sec=0.2

# python Eval_TallyNet.py "Folder_Name"/models/fold"FoldNum" "DatabaseFlag" "FoldNum" "FoldsTotal" "Cut_s" >> "Results_Filename"

# Optional
# "Stride_s" = 1.0 (default)
# "ResampRate" = 15 (default)
# "WinTol" = 0.0 (default) 

python Eval_TallyNet.py Models/"$Folder_Name"/models/fold1 $DatabaseFlag 1 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol > Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
python Eval_TallyNet.py Models/"$Folder_Name"/models/fold2 $DatabaseFlag 2 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
python Eval_TallyNet.py Models/"$Folder_Name"/models/fold3 $DatabaseFlag 3 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
python Eval_TallyNet.py Models/"$Folder_Name"/models/fold4 $DatabaseFlag 4 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
python Eval_TallyNet.py Models/"$Folder_Name"/models/fold0 $DatabaseFlag 5 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt


