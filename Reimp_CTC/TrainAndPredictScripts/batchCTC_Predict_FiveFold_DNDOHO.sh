#!/bin/bash


#SBATCH --job-name Prd_ODND_5F
#SBATCH --output=./joboutput/Results_OHODND_Folds_slurm-%j.out
#SBATCH --error=./joboutput/Results_OHODND_Folds_slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 6gb
#SBATCH --time 3:00:00
#### REMOVING GPU #SBA###TCH --gpus-per-node 1

cd ~/CTC_Rouast_Take2/oneHandOrebaAttempt/



source activate CTC_p306


ModelIdentifier=$1

FoldNum=$2
CheckpointID=$3 # typically model_best_XXXXX or model_latest_YYYYY
EvalID=$4 # ID for to distinguish runs of the Eval, typically the step number from CheckpointID

echo "Evaluating OHO (Both Dom Only) model OHODND_CTC_"$ModelIdentifier" for Fold "$FoldNum" and Checkpoint "$CheckpointID"..."


# Uses OREBA FiveFold split
python main.py --dataset=oreba-one-hand --input_length=512 --seq_shift=8 --model_dir=OHODND_CTC_"$ModelIdentifier"/Fold"$FoldNum" --mode=predict --predict_dir=Preds_OHODND_CTC_"$ModelIdentifier"/Fold"$FoldNum"_"$EvalID" --eval_dir=/project/ahoover/mhealth/jpjolly/CTC_reimp_data/OHODND/Fold"$FoldNum"/eval --model_ckpt="$CheckpointID" --use_def=False --beam_width=3 


# Use the best model checkpoint (drop the .data and .index files)
# seq_length must be 8 from model even though the default is 2
# Eval_dir is used to grab the prediction files for predict mode (even though it is the validation in TrainAndEval Mode)




