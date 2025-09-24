#!/bin/bash


#SBATCH --job-name TN-4SavePred
#SBATCH --output=./joboutput/slurm-%j.out
#SBATCH --error=./joboutput/slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 15gb
#SBATCH --time 6:00:00
#### REMOVING GPU #SBA###TCH --gpus-per-node 1

cd ~/TallyNet_Repo_v2/TallyNet

module load anaconda3/2023.09-0 
module load cuda


source activate tf_v2.2_env


Folder_Name=$1 #Folder Name DOES NOT END IN '/'
DatabaseFlag=$2 # which database to train on; 1=OneHand Oreba, 2=TwoHand Oreba, 3=ClemCafe
DetectMethod=$3 # which placement method to use for TallyNet: midpoint, tetris, midpt tetris, etc.
EvalMethod=$4 # which Evaluation method to use: 1 = PtVsPt, 2 = PtVsWindow, DurationWindow, etc.
Tol=$5 # Tolerance for the detection method if needed
WS=$6
STRIDE_sec=0.2

# python Eval_TallyNet.py "Folder_Name"/models/fold"FoldNum" "DatabaseFlag" "FoldNum" "FoldsTotal" "Cut_s" >> "Results_Filename"

# Optional
# "Stride_s" = 1.0 (default)
# "ResampRate" = 15 (default)
# "WinTol" = 0.0 (default) 

# python Eval_TallyNet.py Models/"$Folder_Name"/models/fold1 $DatabaseFlag 1 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-gts Predictions/GT_"$Folder_Name" > Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py Models/"$Folder_Name"/models/fold2 $DatabaseFlag 2 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-gts Predictions/GT_"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py Models/"$Folder_Name"/models/fold3 $DatabaseFlag 3 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-gts Predictions/GT_"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py Models/"$Folder_Name"/models/fold4 $DatabaseFlag 4 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-gts Predictions/GT_"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py Models/"$Folder_Name"/models/fold0 $DatabaseFlag 5 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-gts Predictions/GT_"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt


### Save Predicitons
# python Eval_TallyNet.py ~/TallyNet_ModelPred_Repo/TallyNet/Models/"$Folder_Name"/models/fold1 $DatabaseFlag 1 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds ~/TallyNet_ModelPred_Repo/TallyNet/Predictions/"$Folder_Name" > Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py ~/TallyNet_ModelPred_Repo/TallyNet/Models/"$Folder_Name"/models/fold2 $DatabaseFlag 2 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds ~/TallyNet_ModelPred_Repo/TallyNet/Predictions/"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py ~/TallyNet_ModelPred_Repo/TallyNet/Models/"$Folder_Name"/models/fold3 $DatabaseFlag 3 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds ~/TallyNet_ModelPred_Repo/TallyNet/Predictions/"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py ~/TallyNet_ModelPred_Repo/TallyNet/Models/"$Folder_Name"/models/fold4 $DatabaseFlag 4 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds ~/TallyNet_ModelPred_Repo/TallyNet/Predictions/"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py ~/TallyNet_ModelPred_Repo/TallyNet/Models/"$Folder_Name"/models/fold0 $DatabaseFlag 5 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds ~/TallyNet_ModelPred_Repo/TallyNet/Predictions/"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt




# #### Save GT AS WELL
# python Eval_TallyNet.py ~/TallyNet_Models_Repo/TallyNet/"$Folder_Name"/models/fold1 $DatabaseFlag 1 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds Predictions/"$Folder_Name" --save-gts /scratch/jpjolly/TallyNetPreds/GT_"$Folder_Name" > Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py ~/TallyNet_Models_Repo/TallyNet/"$Folder_Name"/models/fold2 $DatabaseFlag 2 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds Predictions/"$Folder_Name" --save-gts /scratch/jpjolly/TallyNetPreds/GT_"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py ~/TallyNet_Models_Repo/TallyNet/"$Folder_Name"/models/fold3 $DatabaseFlag 3 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds Predictions/"$Folder_Name" --save-gts /scratch/jpjolly/TallyNetPreds/GT_"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py ~/TallyNet_Models_Repo/TallyNet/"$Folder_Name"/models/fold4 $DatabaseFlag 4 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds Predictions/"$Folder_Name" --save-gts /scratch/jpjolly/TallyNetPreds/GT_"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt
# python Eval_TallyNet.py ~/TallyNet_Models_Repo/TallyNet/"$Folder_Name"/models/fold0 $DatabaseFlag 5 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-preds Predictions/"$Folder_Name" --save-gts /scratch/jpjolly/TallyNetPreds/GT_"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt

#### Save Detection

[ ! -d ~/TallyNet_ModelsDets_Repo/TallyNet/Detections/"$Folder_Name"/ ] && mkdir ~/TallyNet_ModelsDets_Repo/TallyNet/Detections/"$Folder_Name"

python Eval_TallyNet.py ~/TallyNet_ModelsDets_Repo/TallyNet/Models/"$Folder_Name"/models/fold1 $DatabaseFlag 1 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-dets ~/TallyNet_ModelsDets_Repo/TallyNet/Detections/"$Folder_Name"/"$Folder_Name" > Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt

python Eval_TallyNet.py ~/TallyNet_ModelsDets_Repo/TallyNet/Models/"$Folder_Name"/models/fold2 $DatabaseFlag 2 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-dets ~/TallyNet_ModelsDets_Repo/TallyNet/Detections/"$Folder_Name"/"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt

python Eval_TallyNet.py ~/TallyNet_ModelsDets_Repo/TallyNet/Models/"$Folder_Name"/models/fold3 $DatabaseFlag 3 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-dets ~/TallyNet_ModelsDets_Repo/TallyNet/Detections/"$Folder_Name"/"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt

python Eval_TallyNet.py ~/TallyNet_ModelsDets_Repo/TallyNet/Models/"$Folder_Name"/models/fold4 $DatabaseFlag 4 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-dets ~/TallyNet_ModelsDets_Repo/TallyNet/Detections/"$Folder_Name"/"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt

python Eval_TallyNet.py ~/TallyNet_ModelsDets_Repo/TallyNet/Models/"$Folder_Name"/models/fold0 $DatabaseFlag 5 5 $WS --stride $STRIDE_sec --eval-method $EvalMethod --det-method $DetectMethod --tol $Tol --save-dets ~/TallyNet_ModelsDets_Repo/TallyNet/Detections/"$Folder_Name"/"$Folder_Name" >> Results/"$Folder_Name"_d"$DetectMethod"_e"$EvalMethod"_t"$Tol"_Results.txt

