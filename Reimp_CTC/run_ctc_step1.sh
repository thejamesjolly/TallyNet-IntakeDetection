#!/bin/bash

echo "Scheduling Training Jobs for ClemDND data..."

sbatch batchCTC_TrainAndEval_FiveFold_DNDClem.sh 0
sbatch batchCTC_TrainAndEval_FiveFold_DNDClem.sh 1
sbatch batchCTC_TrainAndEval_FiveFold_DNDClem.sh 2
sbatch batchCTC_TrainAndEval_FiveFold_DNDClem.sh 3
sbatch batchCTC_TrainAndEval_FiveFold_DNDClem.sh 4

echo "Completed scheduling jobs."

