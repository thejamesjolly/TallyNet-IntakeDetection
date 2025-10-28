#!/bin/bash


#SBATCH --job-name CTC_DNDCC_TrainModel
#SBATCH --output=./joboutput/DNDClem_Train_FiveFold_slurm-%j.out
#SBATCH --error=./joboutput/DNDClem_Train_FiveFold_slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 8gb
#SBATCH --time 72:00:00
#### REMOVING GPU #SBA###TCH --gpus-per-node 1


cd ~/CTC_Rouast_Take2/REIMP_ctc-intake-TallyNet/

source activate CTC_p306

FoldNum=$1
ModelIdentifier="Example"

echo "Training CTC Model on Clemson data (Dominant and NonDominant GT) fold $FoldNum and Model Name DNDClem_CTC_"$ModelIdentifier"/Fold"$FoldNum"..."
echo "Using adjusted learning rate from OREBA CNN-LSTM paper..."
echo "Using Updated input length size of 120 to match Clemson inertial settings..."

# Uses OREBA recommended split
python main.py --use_def=False --dataset=clemson --input_length=120 --model_dir=DNDClem_CTC_"$ModelIdentifier"/Fold"$FoldNum" --train_epochs=6 --log_steps=500 --eval_steps=500 --lr_base=3e-4 --lr_decay_rate=0.93 --batch_size=256 --train_dir=/project/ahoover/mhealth/jpjolly/CTC_reimp_data/ClemsonDND/Fold"$FoldNum"/train --eval_dir=/project/ahoover/mhealth/jpjolly/CTC_reimp_data/ClemsonDND/Fold"$FoldNum"/valid 

######### NO NEED FOR OUTPUT SINCE IT GOES TO THE JOB OUTPUT FILE ####### > THO_CTC_Fold"$FoldNum"_TrainEval_Output.txt




##### Uses single default meal for quick validation
# python main.py --use_def=False --model_dir=Batch_FullDataModel --predict_dir=Batch_Pred



