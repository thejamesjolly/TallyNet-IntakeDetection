# README FOR TallyNet



TO TRAIN MODELS
===============

Training is done inside CTC Reimplementation GitHub, provided by original authors at the following link:
* https://github.com/prouast/ctc-intake-detection



- NOTE:
	* This repository requires specific version of libraries to operate properly. These are specified in the `CTC_env_libs.txt` file










TO EVALUATE MODELS
==================

To evaluate a model, run the following command in the command line.

`python ReimpCTC_Eval.py [DatasetFlag] [EvalMethod] [Win_tol_sec] [PredFolder]`

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
	
- `Win_tol_sec`
	* Tolerance (in seconds) to allow when matching detections to gesture ground truth window using Kyrit_Pt2Window evaluation

- `PredFold`
	* Name of the folder which contains the results logits from the CTC repositories "--save-preds" option
	* Should end in a `/`

- Full Examples
	* python ReimpCTC_Eval.py 4 1 99 ./FiveFold_CTCResults_Examples/Dom_OHO_Intake/
	* python ReimpCTC_Eval.py 4 2  8 ./FiveFold_CTCResults_Examples/Dom_OHO_Intake/

