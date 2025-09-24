#!/bin/bash



#SBATCH --job-name DOHO_TN-train
#SBATCH --mem 10gb 
#SBATCH --time 12:00:00
#SBATCH --output=./joboutput/OHO_Train_largeWS-slurm-%j.out
#SBATCH --error=./joboutput/OHO_Train_largeWS-slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
##### REMOVE GPU #S##BATCH --gpus=p100:1


#S#B#A#T#C#H --job-name DOHO_TN-train
#S#B#A#T#C#H --mem 15gb 
#S#B#A#T#C#H --time 24:00:00
#S#B#A#T#C#H --output=./joboutput/OHO_Train_largeWS-slurm-%j.out
#S#B#A#T#C#H --error=./joboutput/OHO_Train_largeWS-slurm-%j.out
#S#B#A#T#C#H --nodes 1
#S#B#A#T#C#H --cpus-per-task 2
##### REMOVE GPU #S##BATCH --gpus=p100:1



######  TRAINING ON ONE HAND OREBA DATA

####### MEMORY FOR OREBA OHO
#######      stride=0.5 ==> Mem = 5 GB, time 8 hrs(?)
#######      stride=0.2 ==> Mem = 10, time 15 hrs(?)
#######      WS25, stride=1.0 ==> Mem = 15GB
#######      WS40, stride=1.0 ==> Mem = 25GB



OutputFolder=$1 #Folder Name DOES NOT END IN '/'
Fold=$2 # Current Fold
ModelArchFlag=$3 # Uses selected archetecture 
Cut=$4 # window_size in [sec]

DatabaseFlag=5 # which database to train on; 1=Dnd OneHand Oreba, 2=Dnd TwoHand Oreba, 3 = Dnd Clemson, 4=Dom One Hand OREBA, 5 = Dom Clemson
Stride=0.3 # window stride in [sec]
FoldSplitFlag=1 # 1 = striped fold, 0 = blocked
ResampFlag=15 # set to 15 Hz



cd ~/TallyNet_Experiments/TallyNet_Dom_Models/

module load anaconda3/2023.09-0 
module load cuda

source activate tf_v2.2_env

[ ! -d $OutputFolder ] && mkdir "$OutputFolder"

[ ! -d $OutputFolder/models ] && mkdir "$OutputFolder"/models
[ ! -d $OutputFolder/logs ] && mkdir "$OutputFolder"/logs

echo "running command `python Train_TallyNet.py "$OutputFolder"/models/fold"$Fold" "$Cut" "$Stride" "$DatabaseFlag" "$FoldSplitFlag" "$Fold" 5 "$ModelArchFlag" "$ResampFlag" > "$OutputFolder"/training_fold"$Fold".txt`"

python Train_TallyNet.py "$OutputFolder"/models/fold"$Fold" "$Cut" "$Stride" "$DatabaseFlag" "$FoldSplitFlag" "$Fold" 5 "$ModelArchFlag" "$ResampFlag" > "$OutputFolder"/training_fold"$Fold".txt
