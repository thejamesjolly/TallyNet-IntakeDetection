# Directory README: TallyNet

This directory contains all files needed to reproduce results from the proposed method TallyNet,
proposed by Jolly *et al.*[^TallyNetPaper],
which uses novel training targets of total counts in a long time window 
and a decoding scheme to convert these predicted counts into detections over time.

Before running code, be sure to activate `env1` to load all dependencies.



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
	* Architecture 1333 was used for all models	in the paper

- `ResampFlag`
	* Resampling Rate in Hertz to which all datasets will be interpolated to match.
	* Lower Sampling rate decreases memory and computation, but also model capabilities on harder datasets.
	* 15 Hz is the observed "knee" in F1 vs Sampling Rate where performance is maintained well while reducing complexity 



- Example
	* `python Train_SlidingWindow.py Example_TallyNet_Clem/models/fold4 15 1 3 1 4 5 1333 15 > Example_TallyNet_Clem/training_fold4.txt`











TO EVALUATE MODELS
==================

To evaluate a model, run the following command in the command line.

`python Eval_TallyNet.py [Folder_Name] [DatabaseFlag] [FoldNumber] [TotalFolds] [Cut] [optional-args-pairs]`

- Eval_HeydReimp.py
	* Python Script used for evaluation

- `FolderName`
	* Name of the folder which contains the model .pb file
	* Should be the same as the training folder but with an additional `/models/foldN` appended to it, where N is the fold number
	* Does not end in a `/`

- `DATABASE_Flag`
	* Flag used to select which dataset the model is working with
		- 1 = Dnd OREBA One-Handed
		- 2 = Dnd OREBA Two-Handed
		- 3 = Dnd Clemson
		- 4 = Dom OREBA One-Handed
		- 5 = Dom Clemson

- `FoldNumber`
	* Number of the fold used in k-fold validation (ranging from 1 to `TotalFolds`) used for training and testing split

- `TotalFolds`
	* Total number of folds used in k-fold validation

- `Cut`
	* Total duration of the window used as an input to the model
	* Units are in seconds

- Optional Argument Pairs
	- `--stride [float]`
		* Default = 1.0
		* Step which the dataset will slide the window forward through the motion data to generate the next example
		* Units are in Seconds

	- `--fold-seg [0 or 1]`
		* Default = 1
		* Bool determining whether the k-fold validation is contiguous blocks or striped through all examples
			- 0 = Block Segmentation
			- 1 = Striped Segmentation

	- `--resamp [int]`
		* Default = 15 Hz 
		* Resampling Rate in Hertz to which all datasets will be interpolated to match.
		* Lower Sampling rate decreases memory and computation, but also model capabilities on harder datasets.
		* 15 Hz is the observed "knee" in F1 vs Sampling Rate where performance is maintained well while reducing complexity

	- `--tol [float]`
		* Default = 8.0 sec
		* Tolerance (in seconds) to allow when matching detections to gesture ground truth window using Kyrit_Pt2Window evaluation
		* 8 seconds used as default tolerance to allow for the time most extended gesture sequences may offset the intake moment from the midpoint of the gesture sequence
	
	- `--save-preds [filepath]`
		* Default = Off
		* Saves predictions from evaluation to file prefixed by `filepath`, with a suffix of appended to signify the file as predictions and the corresponding meal number, e.g. `_Pred_MN125.csv`
	
	- `--save-dets [filepath]`
		* Default = Off
		* Saves predictions from evaluation to file prefixed by `filepath`, with a suffix of appended to signify the file as predictions and the corresponding meal number, e.g. `_Pred_MN125.csv`
	
	- `--save-gts [filepath]`
		* Default = Off
		* Saves predictions from evaluation to file prefixed by `filepath`, with a suffix of appended to signify the file as predictions and the corresponding meal number, e.g. `_Pred_MN125.csv`
	
	- `--det-method [int]`
		* Default = 1
		* Toggles decoding method which converts the running predicted counts into specific detections in time
		* "Midpoint Trigger" method used in paper, while others provided and with options to add more.
	
	- `--match-thresh [float]`
		* Default = 0.8
		* Value used as a trigger threshold in some detection methods
	
	- `--eval-method [int]`
		* Defualt = 2 (Point to Window with tolerance)
		* Flag to select which evaluation method to use matching detections to GT labels.
			* 1 = Point to Point evaluation
			* 2 = Point to Window evaluation
		* Additional values could be extended to include additional evaluation methods, such as Window to Window duration evaluation
		


- Full Example
	* `python Eval_TallyNet.py TallyNet_Example_THO/models/fold4 2 4 5 15 --stride 1.5 --tol 10 >> Example_THO_Results.txt`


## References
[^TallyNetPaper]: TBD TallyNet Reference
