#!/bin/bash

#SBATCH --job-name THO_TN-train
#SBATCH --mem 30gb 
#SBATCH --time 26:00:00
#SBATCH --output=./joboutput/THO_train_largeWS_slurm-%j.out
#SBATCH --error=./joboutput/THO_train_largeWS_slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
##### REMOVE GPU #S##BATCH --gpus=p100:1



######  TRAINING ON TWO HAND OREBA DATA

####### MEMORY FOR OREBA THO
#######      stride=0.5 ==> Mem = 8 GB, time 8hrs
#######      stride=0.2 ==> Mem = 30 GB (Maybe 25?)
#######      WS25, stride=1.0 ==> Mem = 20GB
#######      WS25, stride=1.0 ==> Mem = 40GB



OutputFolder=$1 #Folder Name DOES NOT END IN '/'
Fold=$2 # Current Fold
ModelArchFlag=$3 # Uses selected archetecture 
Cut=$4 # window_size in [sec]

DatabaseFlag=2 # which database to train on; 1=Dnd OneHand Oreba, 2=Dnd TwoHand Oreba, 3 = Dnd Clemson, 4=Dom One Hand OREBA, 5 = Dom Clemson
Stride=0.3 # window stride in [sec]
FoldSplitFlag=1 # 1 = striped fold, 0 = blocked
ResampFlag=15 # set to 15 Hz



cd ~/TallyNet_Experiments/TallyNet_Paper_Models/

module load anaconda3/2023.09-0 
module load cuda

source activate tf_v2.2_env

[ ! -d $OutputFolder ] && mkdir "$OutputFolder"

[ ! -d $OutputFolder/models ] && mkdir "$OutputFolder"/models
[ ! -d $OutputFolder/logs ] && mkdir "$OutputFolder"/logs

echo "running command `python Train_TallyNet.py "$OutputFolder"/models/fold"$Fold" "$Cut" "$Stride" "$DatabaseFlag" "$FoldSplitFlag" "$Fold" 5 "$ModelArchFlag" "$ResampFlag" > "$OutputFolder"/training_fold"$Fold".txt`"

python Train_TallyNet.py "$OutputFolder"/models/fold"$Fold" "$Cut" "$Stride" "$DatabaseFlag" "$FoldSplitFlag" "$Fold" 5 "$ModelArchFlag" "$ResampFlag" > "$OutputFolder"/training_fold"$Fold".txt
