#!/bin/bash


#SBATCH --job-name TallyNet-Eval
#SBATCH --output=./joboutput/slurm-%j.out
#SBATCH --error=./joboutput/slurm-%j.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem 12gb
#SBATCH --time 4:00:00
#SBATCH --gpus-per-node 1

cd ~/TallyNet_Experiments/TallyNet_Models

module load anaconda3/2023.09-0 
module load cuda


source activate tf_v2.2_env



# python Eval_TallyNet.py "Folder_Name"/models/fold1 1 5 "FoldSplitFlag" "Cut_s" "Stride_s" "DatabaseFlag" "ResampRate" "WinTol" >> "Results_Filename"


# python Eval_TallyNet.py Models/Clem_0/models/fold1 1 5 1 15 1 3 15 12 > Results/Clem_0_Results.txt
# python Eval_TallyNet.py Models/Clem_0/models/fold2 2 5 1 15 1 3 15 12 >> Results/Clem_0_Results.txt
# python Eval_TallyNet.py Models/Clem_0/models/fold3 3 5 1 15 1 3 15 12 >> Results/Clem_0_Results.txt
# python Eval_TallyNet.py Models/Clem_0/models/fold4 4 5 1 15 1 3 15 12 >> Results/Clem_0_Results.txt
# python Eval_TallyNet.py Models/Clem_0/models/fold0 5 5 1 15 1 3 15 12 >> Results/Clem_0_Results.txt

# python Eval_TallyNet.py Models/OHO_0/models/fold1 1 5 1 15 1 1 15 12 > Results/OHO_0_Results.txt
# python Eval_TallyNet.py Models/OHO_0/models/fold2 2 5 1 15 1 1 15 12 >> Results/OHO_0_Results.txt
# python Eval_TallyNet.py Models/OHO_0/models/fold3 3 5 1 15 1 1 15 12 >> Results/OHO_0_Results.txt
# python Eval_TallyNet.py Models/OHO_0/models/fold4 4 5 1 15 1 1 15 12 >> Results/OHO_0_Results.txt
# python Eval_TallyNet.py Models/OHO_0/models/fold0 5 5 1 15 1 1 15 12 >> Results/OHO_0_Results.txt

# python Eval_TallyNet.py Models/THO_0/models/fold1 1 5 1 15 1 2 15 12 > Results/THO_0_Results.txt
# python Eval_TallyNet.py Models/THO_0/models/fold2 2 5 1 15 1 2 15 12 >> Results/THO_0_Results.txt
# python Eval_TallyNet.py Models/THO_0/models/fold3 3 5 1 15 1 2 15 12 >> Results/THO_0_Results.txt
# python Eval_TallyNet.py Models/THO_0/models/fold4 4 5 1 15 1 2 15 12 >> Results/THO_0_Results.txt
# python Eval_TallyNet.py Models/THO_0/models/fold0 5 5 1 15 1 2 15 12 >> Results/THO_0_Results.txt


python Eval_TallyNet.py Models/OHO_A101_1/models/fold1 1 5 1 15 1 1 15 12 > Results/OHO_A101_0_Results.txt
python Eval_TallyNet.py Models/OHO_A101_1/models/fold2 2 5 1 15 1 1 15 12 >> Results/OHO_A101_0_Results.txt
python Eval_TallyNet.py Models/OHO_A101_1/models/fold3 3 5 1 15 1 1 15 12 >> Results/OHO_A101_0_Results.txt
python Eval_TallyNet.py Models/OHO_A101_1/models/fold4 4 5 1 15 1 1 15 12 >> Results/OHO_A101_0_Results.txt
python Eval_TallyNet.py Models/OHO_A101_1/models/fold0 5 5 1 15 1 1 15 12 >> Results/OHO_A101_0_Results.txt

