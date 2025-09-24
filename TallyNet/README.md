# README FOR TallyNet



TO TRAIN MODELS
===============

To train a model, use the following prompt in the command line.

`python Train_TallyNet.py [FolderName] [Cut] [Stride] [DATABASE_Flag] [FoldSegFlag] [FoldNumber] [TotalFolds] [ModelArchFlag] [ResampFlag]`

- Train_TallyNet.py
	* Python Script to train TallyNet model

- `FolderName`
	* Output folder location to store all files generated with the specific fold and model currently training

- `Cut`
	* Total duration of the window used as an input to the model
	* Units are in seconds

- `Stride`
	* Step which the dataset will slide the window forward through the motion data to generate the next example
	* Units are in Seconds

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

- `ModelArchFlag`
	* Model Architecture ID based to ModelSelect.py to return hidden layer structure of the model

- `ResampFlag`
	* Resampling Rate in Hertz to which all datasets will be interpolated to match.
	* Lower Sampling rate decreases memory and computation, but also model capabilities on harder datasets.
	* 15 Hz is the observed "knee" in F1 vs Sampling Rate where performance is maintained well while reducing complexity 



- Example
	* `python Train_SlidingWindow.py Example_TallyNet_Clem/models/fold4 15 1 3 1 4 5 101 15 > Example_TallyNet_Clem/training_fold4.txt`











TO EVALUATE MODELS
==================

To evaluate a model, run the following command in the command line.

`python Eval_TallyNet.py [Folder_Name] [FoldNumber] [TotalFolds] [FoldSplitFlag] [Cut] [Stride] [DatabaseFlag] [ResampRate] [WinTol]`

- Eval_HeydReimp.py
	* Python Script used for evaluation

- `FolderName`
	* Name of the folder which contains the model .pb file
	* Should be the same as the training folder but with an additional `/models/foldN` appended to it, where N is the fold number
	* Does not end in a `/`

- `FoldNumber`
	* Number of the fold used in k-fold validation (ranging from 1 to `TotalFolds`) used for training and testing split

- `TotalFolds`
	* Total number of folds used in k-fold validation

- `FoldSegFlag`
	* Bool determining whether the k-fold validation is contiguous blocks or striped through all examples
		- 0 = Block Segmentation
		- 1 = Striped Segmentation

- `Cut`
	* Total duration of the window used as an input to the model
	* Units are in seconds

- `Stride`
	* Step which the dataset will slide the window forward through the motion data to generate the next example
	* Units are in Seconds

- `DATABASE_Flag`
	* Flag used to select which dataset the model is working with
		- 1 = Dnd OREBA One-Handed
		- 2 = Dnd OREBA Two-Handed
		- 3 = Dnd Clemson
		- 4 = Dom OREBA One-Handed
		- 5 = Dom Clemson

- `ResampFlag`
	* Resampling Rate in Hertz to which all datasets will be interpolated to match.
	* Lower Sampling rate decreases memory and computation, but also model capabilities on harder datasets.
	* 15 Hz is the observed "knee" in F1 vs Sampling Rate where performance is maintained well while reducing complexity

- `Win_tol_sec`
	* Tolerance (in seconds) to allow when matching detections to gesture ground truth window using Kyrit_Pt2Window evaluation


- Full Example
	* `python Eval_TallyNet.py TallyNet_Example_THO/models/fold4 4 5 1 15 1 2 15 12 >> Example_THO_Results.txt`
