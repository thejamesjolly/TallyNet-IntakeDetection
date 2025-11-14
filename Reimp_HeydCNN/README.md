# Directory README: HeydReimp_Comp

This directory contains all files needed to reproduce results from the binary classifier benchmark method,
which was orignally proposed by Heydarian *et al.*[^HeydCNNPaper].
This method uses a CNN-LSTM model with 2 seconds of input data to predict probabilities
of whether the input is an intake gesture or not,
with these probabilities post-processed into detections of intake over the meal.

Some example pre-trained models from the paper are provided in the `Models` directory.
Final detections for these example models are provided in `Detections` directory.
Tuning the threshold used to trigger detections to balance TPR and PPV is necessary,
and the range of values used are shown in the `PAPER_RECORD_script_HeydCNN_Eval` script.
The `Results` directories is empty, 
but are used to output results from the evaluation script.

Before running code, be sure to activate `env1` to load all dependencies.

## TO TRAIN MODELS

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











## TO EVALUATE MODELS

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



## References

[^HeydCNNPaper]: H. Heydarian, P. V. Rouast, M. T. P. Adam, T. Burrows, C. E. Collins and M. E. Rollo,
"Deep Learning for Intake Gesture Detection From Wrist-Worn Inertial Sensors: 
The Effects of Data Preprocessing, Sensor Modalities, and Sensor Positions," 
in IEEE Access, vol. 8, pp. 164936-164949, 2020, doi: 10.1109/ACCESS.2020.3022042.
keywords: {Sensors;Machine learning;Data models;Data preprocessing;Accelerometers;Gyroscopes;Hidden Markov models;Accelerometer;deep learning;intake gesture detection;gyroscope;wrist-worn}
