#!/bin/bash

#SBATCH --job-name tallyNet-training
#SBATCH --output=./joboutput/slurm-%j.out
#SBATCH --error=./joboutput/slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 40gb
#SBATCH --time 20:00:00
#SBATCH --gpus-per-node 1


OutputFolder="Models/TallyNet_Example_OHO" #Folder Name DOES NOT END IN '/'
Cut=15 # window_size in [sec]
Stride=2 # window stride in [sec]
DatabaseFlag=1 # which database to train on; 1=OneHand Oreba, 2=TwoHand Oreba, 3=ClemCafe
FoldSplitFlag=1 # 1 = striped fold, 0 = blocked
ModelArchFlag=101 # Uses selected archetecture 
ResampFlag=15 # set to 15 Hz

# Current Fold
Fold=$1
echo "Running TallyNet Training for One-Hand OREBA fold ($Fold)!"

cd ~/TallyNet_Repo/TallyNet_Models/

module load anaconda3/2023.09-0 
module load cuda

source activate tf_v2.2_env

[ ! -d $OutputFolder ] && mkdir "$OutputFolder"

[ ! -d $OutputFolder/models ] && mkdir "$OutputFolder"/models
[ ! -d $OutputFolder/logs ] && mkdir "$OutputFolder"/logs

python Train_SlidingWindow.py "$OutputFolder"/models/fold"$Fold" "$Cut" "$Stride" "$DatabaseFlag" "$FoldSplitFlag" "$Fold" 5 "$ModelArchFlag" "$ResampFlag" > "$OutputFolder"/training_fold"$Fold".txt
