#!/bin/bash


#SBATCH --job-name Prd_TH5F
#SBATCH --output=./joboutput/Results_THO_Folds_slurm-%j.out
#SBATCH --error=./joboutput/Results_THO_Folds_slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 6gb
#SBATCH --time 3:00:00
#### REMOVING GPU #SBA###TCH --gpus-per-node 1

cd ~/CTC_Rouast_Take2/ctc-intake-detection-master/



source activate CTC_p306

if [[ $# -ne 4 ]]; then
  echo "Illegal number of parameters:"
  echo "   Expected Usage: [Fold Num] [CheckpointName] [EvalID]"
  exit 64
fi


ModelIdentifier=$1

FoldNum=$2
CheckpointID=$3 # typically model_best_XXXXX or model_latest_YYYYY
EvalID=$4 # ID for to distinguish runs of the Eval, typically the step number from CheckpointID

echo "Evaluating THO model Fold "$FoldNum" and Checkpoint "$CheckpointID"..."

# Uses OREBA recommended split
python main.py --dataset=oreba-dis --input_length=512 --seq_shift=8 --model_dir=THO_CTC_"$ModelIdentifier"/Fold"$FoldNum" --mode=predict --predict_dir=Preds_THO_"$ModelIdentifier"/Fold"$FoldNum"_"$EvalID" --eval_dir=/scratch/jpjolly/CTC_reimp_data/OREBA/Fold"$FoldNum"/eval --model_ckpt="$CheckpointID" --use_def=False --beam_width=3 


# Use the best model checkpoint (drop the .data and .index files)
# seq_length must be 8 from model even though the default is 2
# Eval_dir is used to grab the prediction files for predict mode (even though it is the validation in TrainAndEval Mode)




