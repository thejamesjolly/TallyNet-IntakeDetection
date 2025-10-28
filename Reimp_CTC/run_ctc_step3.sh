#!/bin/bash


# """
# Find all best running checkpoints from step2 and run this command with the ^Filler^ value replaced with the number of the checkpoint; e.g. 41500 in model_best_41500
# """


echo "Scheduling Eval Jobs for ClemDND data..."

sbatch batchCTC_Predict_FiveFold_DNDClem.sh 0 model_best_^F0FILLER^ B^F0FILLER^
sbatch batchCTC_Predict_FiveFold_DNDClem.sh 1 model_best_^F1FILLER^ B^F1FILLER^
sbatch batchCTC_Predict_FiveFold_DNDClem.sh 2 model_best_^F2FILLER^ B^F2FILLER^
sbatch batchCTC_Predict_FiveFold_DNDClem.sh 3 model_best_^F3FILLER^ B^F3FILLER^
sbatch batchCTC_Predict_FiveFold_DNDClem.sh 4 model_best_^F4FILLER^ B^F4FILLER^

echo "Completed scheduling jobs."
