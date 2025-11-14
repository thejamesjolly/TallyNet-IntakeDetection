# Directory README: Reimp_CTC

This directory contains all files needed to reproduce results from the CTC sequence benchmark method,
which was orignally proposed by Rouast *et al.*[^RouastCTCPaper].
This method uses CTC loss neural network and 8 second inputs to produce 
a sequence of class proabilities (intake/non-intake, or bite/drink/non-intake),
and decodes these probabilities wtih an extended beam search to produce detections.

Before running code, be sure to activate `env2` to load all dependencies. 
This environment is specific to the CTC Sequence benchmark.

Much of this benchmark code is provided by the orignal authors in three GitHub Repositories.
These repositories have been extended here to allow for processing 
of the additional dataset variations, such as the One Handed OREBA or the Clemson-DND data.

Orignal Source Repositories can be found at the following links:
- (CTC Data Processing)[https://github.com/prouast/inertial-sensor-processing]
- (CTC TensorFlow Custom Layer)[https://github.com/prouast/ctc-beam-search-op]
- (CTC Intake Detection)[https://github.com/prouast/ctc-intake-detection/]

## Example Models and Detections

Some example pre-trained models from the paper are provided in the `Models` directory.
Final detections for these example models are provided in `Detections` directory.

Additionanlly, before getting depenency version issues corrected, 
the original authors were kind enough to provide 
their results for the orginal paper[^RouastCTCPaper],
which was run on a specified train/valid/eval split 
for the DndTHO and DomClem datasets.
These detections files are found in the `Detections_ProvidedByAuthors` directory.  


## Data Processing and Custom Layer Creation

Before running model code, data must be prcoessed into TF_Records 
and a custom Tensorflow layer for the extended beam search operation must be created.

Follow instructions in `ctc-beam-search-op-TallyNetUpdate/` to create 
a python pip wheel with the custom layer, 
which can then be installed and added to `env2`.

Next, follow instructions in `inertial-sensor-processing-TallyNet/` to create
the TF_Records for each dataset.
These files will be created for all participants in the dataset,
and must then be organized into separate directories 
for training, validation, and testing splits.
The original authors used one split for results,
while we extended this to 5-fold validation on each dataset.
To generate the 5 folds, all 100 records can be split
using their index modulo 5 to filter into evaluation folders.
Validation is selected as the preceeding fold number 
(e.g. fold 3 validation comes from fold 2,
fold 0 validation comes from fold 4, etc.).
Training splits are selected as the validation records 
from the succeeding 3 folds
(e.g. fold 1 training comes from folds 2, 3, and 4 evaluation records).

While this duplication uses additional memory, 
this structure keeps the original authors' data processing code.
If memory constraints are a concern, modifications could allow
for dynmaic splitting of the training and evaluation records.



## Usage

### Training and Prediction

To produce detections from the model, follow the three step example provided in the files
`run_ctc_step1.sh`, `run_ctc_step2.sh`, and `run_ctc_step3.sh` provided in this directory.
These scripts should be run inside the directory `ctc-intake-detectionTallyNetUpdate/`.
These files are set up to use a slurm scheduler, and could be modified to run locally 
or with a different scheduler.
If using these scripts directly, all script files from `TrainAndPredictScripts/` 
should be moved into this directory.

Step 1 is to train the models, requiring a training to be spawned for each fold of each dataset.
Either select the training script to match the desired dataset file,
or use the commands provided in the `batchCTC_TrainAndEval_FiveFold_DATASET.sh` file 
to run the code in training mode. 
It is important that training output is writtten to a file.

After training, Step 2 scans through the training file output 
to determine which of the benchmarks has teh best validation performance.
The script provided will pull out the validation results, 
and the user should note the filename (i.e. checkmark number)
with the highest F1 score.

Finally, Step 3 needs to be updated with the top performing checkpoint,
then run to produce the detections.
This will output a folder with all of the prediction logits.
These should be moved into a directory inside `Evaluation/FiveFold_CTCResults_Examples/`
matching the 5 fold format of the provided examples. 


### Evaluation

To produce final performance metrics, instructions are provided
inside the `README.md` file of the `Evaluation` directory.


## References

[^RouastCTCPaper]: P. V. Rouast and M. T. P. Adam, 
"Single-Stage Intake Gesture Detection Using CTC Loss
and Extended Prefix Beam Search,"
in IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 7, pp. 2733-2743, July 2021, 
doi: 10.1109/JBHI.2020.3046613.
keywords: {Decoding;Monitoring;Training;Task analysis;Timing;Estimation;Biological system modeling;CTC;deep learning;dietary monitoring;inertial and video sensors;intake gesture detection}

