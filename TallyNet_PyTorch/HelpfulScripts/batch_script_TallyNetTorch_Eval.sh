#!/bin/bash


#SBATCH --job-name TN-Eval
#SBATCH --output=./joboutput/TN_Ev_slurm-%j.out
#SBATCH --error=./joboutput/TN_Ev_slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16gb
#SBATCH --time 3:00:00
#SBATCH --gpus=1
### REMOVING GPU #S#BATCH --gpus-per-node 1

cd ~/TallyNet_Repo_v2/TallyNet_Torch

module load anaconda3/2023.09-0 
module load cuda


source activate PytorchWorkshop


Folder_Name=$1 #Folder Name DOES NOT END IN '/'
WindowSizeSec=$2
DatabaseFlag=$3 # which database to train on; 1=OneHand Oreba, 2=TwoHand Oreba, 3=ClemCafe
DetectMethod=$4 # which placement method to use for TallyNet: midpoint, tetris, midpt tetris, etc.
EvalMethod=$5 # which Evaluation method to use: 1 = PtVsPt, 2 = PtVsWindow, DurationWindow, etc.
Tol=$6 # Tolerance for the detection method if needed
ModelArchFlag=$7 # Architecture select value to use



STRIDE_sec=0.2

# python Eval_TallyNet.py "Folder_Name"/models/fold"FoldNum" "DatabaseFlag" "FoldNum" "FoldsTotal" "Cut_s" >> "Results_Filename"

# Optional
# "Stride_s" = 1.0 (default)
# "ResampRate" = 15 (default)
# "WinTol" = 0.0 (default) 

python Torch_Eval_TallyNet.py Models/"$Folder_Name"/models/fold1_final.pt $DatabaseFlag 1 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --arch $ModelArchFlag > Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
python Torch_Eval_TallyNet.py Models/"$Folder_Name"/models/fold2_final.pt $DatabaseFlag 2 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --arch $ModelArchFlag >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
python Torch_Eval_TallyNet.py Models/"$Folder_Name"/models/fold3_final.pt $DatabaseFlag 3 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --arch $ModelArchFlag >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
python Torch_Eval_TallyNet.py Models/"$Folder_Name"/models/fold4_final.pt $DatabaseFlag 4 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --arch $ModelArchFlag >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
python Torch_Eval_TallyNet.py Models/"$Folder_Name"/models/fold0_final.pt $DatabaseFlag 5 5 $WindowSizeSec --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --arch $ModelArchFlag >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt


