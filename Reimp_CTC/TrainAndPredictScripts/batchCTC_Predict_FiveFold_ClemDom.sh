#!/bin/bash


#SBATCH --job-name Prd_C5F
#SBATCH --output=./joboutput/Results_Clem_Folds_slurm-%j.out
#SBATCH --error=./joboutput/Results_Clem_Folds_slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 8gb
#SBATCH --time 12:00:00
#### REMOVING GPU #SBA###TCH --gpus-per-node 1

cd ~/CTC_Rouast_Take2/ctc-intake-detection-master/



source activate CTC_p306

if [[ $# -ne 4 ]]; then
  echo "Illegal number of parameters:"
  echo "   Expected Usage: [ModelID] [Fold Num] [CheckpointName] [EvalID]"
  exit 64
fi

ModelIdentifier=$1

FoldNum=$2
CheckpointID=$3 # typically model_best_XXXXX or model_latest_YYYYY
EvalID=$4 # ID for to distinguish runs of the Eval, typically the step number from CheckpointID

echo "Evaluating Clemson model Fold "$FoldNum" and Checkpoint "$CheckpointID"..."

# Uses OREBA recommended split
python main.py --dataset=clemson --input_length=120 --seq_shift=2 --model_dir=ClemDom_CTC_"$ModelIdentifier"/Fold"$FoldNum" --mode=predict --predict_dir=Preds_ClemDom_"$ModelIdentifier"/Fold"$FoldNum"_"$EvalID" --eval_dir=/project/ahoover/mhealth/jpjolly/CTC_reimp_data/ClemsonDom/Fold"$FoldNum"/eval --model_ckpt="$CheckpointID" --use_def=False --beam_width=3 


# Use the best model checkpoint (drop the .data and .index files)
# seq_length must be 8 from model even though the default is 2
# Eval_dir is used to grab the prediction files for predict mode (even though it is the validation in TrainAndEval Mode)




