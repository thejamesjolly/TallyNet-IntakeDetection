# README FOR TallyNet



## TO TRAIN MODELS


Training is done inside CTC Reimplementation GitHub, provided by original authors at the following link:
* https://github.com/prouast/ctc-intake-detection

Modifications to this code are provided in the corresponding directory within `Reimp_CTC/`.



### NOTE:
This repository requires specific version of libraries to operate properly. These are specified in the `requirements_env2.txt` file found in the TallyNet Repositories main directory.


## Prediction Preparations

Before running these evaluation scripts to calculate performance,
all predictions from the training and prediction stage 
should be stored in a folder matching the formatting of the examples in `FiveFold_CTCResults_Examples/`.
This filepath will be used to run the evaluation scripts as the `PredFolder` input.

## TO EVALUATE MODELS

To evaluate a model, run the following command in the command line.

`python ReimpCTC_Eval.py [DatasetFlag] [EvalMethod] [WinTolSec] [PredFolder]`

- ReimpCTC_Eval.py
	* Python Script used for evaluation

- `DATABASE_Flag`
	* Flag used to select which dataset the model is working with
		- 1 = Dnd OREBA One-Handed
		- 2 = Dnd OREBA Two-Handed
		- 3 = Dnd Clemson
		- 4 = Dom OREBA One-Handed
		- 5 = Dom Clemson
- `EvalMethod`
	* Flag used to select which evaluation method to use to match detections and GT events
		- 1 = Point to Point Eval
		- 2 = Point to Window with tolerance Eval
	
- `WinTolSec`
	* Tolerance (in seconds) to allow when matching detections to gesture ground truth window using Kyrit_Pt2Window evaluation

- `PredFolder`
	* Name of the folder which contains the results logits from the CTC repositories "--save-preds" option
	* Should end in a `/`

- Full Examples
	* python ReimpCTC_Eval.py 4 1 99 ./FiveFold_CTCResults_Examples/Dom_OHO_Intake/
	* python ReimpCTC_Eval.py 4 2  8 ./FiveFold_CTCResults_Examples/Dom_OHO_Intake/

