#!/bin/bash


#SBATCH --job-name DomCleanClem_TN-train
#SBATCH --mem 32gb 
#SBATCH --time 4:00:00
#SBATCH --output=./joboutput/DomClem_Train-slurm-%j.out
#SBATCH --error=./joboutput/DomClem_Train-slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --gpus=1



#S%B%%A%T%C%H --job-name DCleanClem_TN-train
#S%B%%A%T%C%H --mem 24gb 
#S%B%%A%T%C%H --time 36:00:00
#S%B%%A%T%C%H --output=./joboutput/CleanClem_Train_smallWS-slurm-%j.out
#S%B%%A%T%C%H --error=./joboutput/CleanClem_Train_smallWS-slurm-%j.out
#S%B%%A%T%C%H --nodes 1
#S%B%%A%T%C%H --cpus-per-task 2
##### REMOVE GPU #S##BATCH --gpus=p100:1





######  TRAINING ON CLEAN CLEMSON DATA (with Gesture Labels) DATA

####### MEMORY FOR CLEMSON
#######      stride=0.5 ==> Mem = 24 GB  time (maybe more) 36hrs
#######      stride=0.2 ==> Mem = 60 GB
#######      WS25, stride=1.0 ==> Mem = 48GB
#######      WS40, stride=1.0 ==> Mem = 64GB




OutputFolder=$1 #Folder Name DOES NOT END IN '/'
Fold=$2 # Current Fold
ModelArchFlag=$3 # Uses selected archetecture 
Cut=$4 # window_size in [sec]

DatabaseFlag=5 # which database to train on; 1=Dnd OneHand Oreba, 2=Dnd TwoHand Oreba, 3 = Dnd Clemson, 4=Dom One Hand OREBA, 5 = Dom Clemson
Stride=0.5 # window stride in [sec]
FoldSplitFlag=1 # 1 = striped fold, 0 = blocked
ResampFlag=15 # set to 15 Hz



cd ~/TallyNet_Repo_v2/TallyNet_Torch/

module load anaconda3/2023.09-0 
module load cuda

source activate PytorchWorkshop

[ ! -d $OutputFolder ] && mkdir "$OutputFolder"

[ ! -d $OutputFolder/models ] && mkdir "$OutputFolder"/models
[ ! -d $OutputFolder/logs ] && mkdir "$OutputFolder"/logs

echo "running command: python Train_TallyNet.py "$OutputFolder"/models/fold"$Fold" "$Cut" "$Stride" "$DatabaseFlag" "$FoldSplitFlag" "$Fold" 5 "$ModelArchFlag" "$ResampFlag" > "$OutputFolder"/training_fold"$Fold".txt"

python Torch_Train_TallyNet.py "$OutputFolder"/models/fold"$Fold" "$Cut" "$Stride" "$DatabaseFlag" "$FoldSplitFlag" "$Fold" 5 "$ModelArchFlag" "$ResampFlag" > "$OutputFolder"/training_fold"$Fold".txt
