#!/bin/bash


#SBATCH --job-name Prd_DomOHO_5F
#SBATCH --output=./joboutput/Results_DomOHO_Folds_slurm-%j.out
#SBATCH --error=./joboutput/Results_DomOHO_Folds_slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 6gb
#SBATCH --time 3:00:00
#### REMOVING GPU #SBA###TCH --gpus-per-node 1

cd ~/CTC_Rouast_Take2/oneHandOrebaAttempt/



source activate CTC_p306

if [[ $# -ne 4 ]]; then
  echo "Illegal number of parameters:"
  echo "   Expected Usage: [Fold Num] [CheckpointName] [EvalID]"
  exit 64
fi


FoldNum=$1
CheckpointID=$2 # typically model_best_XXXXX or model_latest_YYYYY
EvalID=$3 # ID for to distinguish runs of the Eval, typically the step number from CheckpointID
ModelIdentifier=$4

echo "Evaluating OHODom (Dom Only GT) model "$ModelIdentifier" Fold "$FoldNum" and Checkpoint "$CheckpointID"..."

# Uses OREBA recommended split
python main.py --dataset=oreba-one-hand --input_length=512 --seq_shift=8 --model_dir=OHODom_CTC_"$ModelIdentifier"/Fold"$FoldNum" --mode=predict --predict_dir=Preds_OHODom_CTC_"$ModelIdentifier"/Fold"$FoldNum"_"$EvalID" --eval_dir=/project/ahoover/mhealth/jpjolly/CTC_reimp_data/OHODom/Fold"$FoldNum"/eval --model_ckpt="$CheckpointID" --use_def=False --beam_width=3 


# Use the best model checkpoint (drop the .data and .index files)
# seq_length must be 8 from model even though the default is 2
# Eval_dir is used to grab the prediction files for predict mode (even though it is the validation in TrainAndEval Mode)




