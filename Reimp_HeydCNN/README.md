# README FOR HeydReimp_Comp



TO TRAIN MODELS
===============

To train a model, use the following prompt in the command line.

`python Train_HeydCNN.py [FolderName] [DATABASE_Flag]  [FoldSegFlag] [FoldNumber] [TotalFolds]`

- Train_OREBA.py
	* Python Script to train model

- `FolderName`
	* Output folder location to store all files generated with the specific fold and model currently training

- `DATABASE_Flag` 
	* Flag used to select which dataset the model is working with
		- 1 = Dnd OREBA One-Handed
		- 2 = Dnd OREBA Two-Handed
		- 3 = Dnd Clemson
		- 4 = Dom OREBA One-Handed
		- 5 = Dom Clemson

- `FoldSegFlag`
	* Bool determining whether the k-fold validation is contiguous blocks or striped through all examples
		- 0 = Block Segmentation
		- 1 = Striped Segmentation

- `FoldNumber`
	* Number of the fold used in k-fold validation (ranging from 1 to `TotalFolds`) used for training and testing split

- `TotalFolds`
	* Total number of folds used in k-fold validation

- Example
	* `python Train_OREBA.py "$OutputFolder"/models/fold"$Fold" "$DatabaseFlag" "$FoldSplitFlag" "$Fold" 5 > "$OutputFolder"/training_fold"$Fold".txt`











TO EVALUATE MODELS
==================

To evaluate a model, run the following command in the command line.

`python Eval_HeydCNN.py [FolderName] [FoldNumber] [TotalFolds] [FoldSegFlag] [STR_sec] [DATABASE_Flag] [Win_tol_sec] `

- Eval_HeydReimp.py
	* Python Script used for evaluation

- `FolderName`
	* Name of the folder generated at training, which contains the model .pb file  


- `FoldNumber`
	* Number of the fold used in k-fold validation (ranging from 1 to `TotalFolds`) used for training and testing split

- `TotalFolds`
	* Total number of folds used in k-fold validation


- `FoldSegFlag`
	* Bool determining whether the k-fold validation is contiguous blocks or striped through all examples
		- 0 = Block Segmentation
		- 1 = Striped Segmentation

- `STR_sec`
	* Duration in seconds of the stride to move the evaluation model forward

- `DATABASE_Flag`
	* Flag used to select which dataset the model is working with
		- 1 = Dnd OREBA One-Handed
		- 2 = Dnd OREBA Two-Handed
		- 3 = Dnd Clemson
		- 4 = Dom OREBA One-Handed
		- 5 = Dom Clemson

- `Win_tol_sec`
	* Tolerance (in seconds) to allow when matching detections to gesture ground truth window using Kyrit_Pt2Window evaluation


- Full Example
	* `python Eval_HeydReimp.py Example_HeydCNN_Clem/models/fold3 3 5 1 1 3 12 >> "$Results_Filename"`
