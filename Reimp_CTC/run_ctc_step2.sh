#!/bin/bash

# """
# Find all best running Evaluation scores by running this command with the %j replaced by the corresponding job ID from the prior training runs.
# """



echo -e "Fold0 results...\n"
grep Keeping joboutput/DNDClem_Train_FiveFold_slurm-%j.out
echo -e "\n\n"

echo -e "Fold1 results...\n"
grep Keeping joboutput/DNDClem_Train_FiveFold_slurm-%j.out
echo -e "\n\n"

echo -e "Fold2 results...\n"
grep Keeping joboutput/DNDClem_Train_FiveFold_slurm-%j.out
echo -e "\n\n"

echo -e "Fold3 results...\n"
grep Keeping joboutput/DNDClem_Train_FiveFold_slurm-%j.out
echo -e "\n\n"

echo -e "Fold4 results...\n"
grep Keeping joboutput/DNDClem_Train_FiveFold_slurm-%j.out
echo -e "\n\n"


