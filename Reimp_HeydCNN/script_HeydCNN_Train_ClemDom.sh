#!/bin/bash

#SBATCH --job-name DClem_HeydCNN-Train
#SBATCH --output=./joboutput/DomClem_Train-slurm-%j.out
#SBATCH --error=./joboutput/DomClem_Train-slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 5gb
#SBATCH --time 5:00:00
#SB####ATCH --gpus-per-node 1

cd ~/TallyNet_Experiments/Reimp_HeydCNN

module load anaconda3/2023.09-0 
module load cuda

source activate tf_v2.2_env

FoldIdx=$1
FoldsTotal=5
$ModelIdentifier="4"

echo "FoldTotal = "$FoldsTotal" and FoldIdx = "$FoldIdx""

# python Train_HeydCNN ModelName FoldIdx FoldTotal DatasetFlag


# DomClem
# echo "Running With CPU"
echo "Running Command: python Train_HeydCNN.py DomClem_HeydCNN_"$ModelIdentifier"/models/fold"$FoldsIdx" "$FoldIdx" "$FoldsTotal" 5"
python Train_HeydCNN.py "DomClem_HeydCNN_"$ModelIdentifier"/models/fold"$FoldIdx"" "$FoldIdx" "$FoldsTotal" 5

