# TallyNet_IntakeDetection

Codebase for TallyNet Paper[^TallyNetPaper]. This repository includes the implementation of TallyNet, database preprocessing for intake detection datasets, and benchmark reimplementations (Dong *et al.*[^DongHeurPaper], Heydarian *et al.*[^HeydCNNPaper], Rouast *et al.*[^RouastCTCPaper]).



## Repository Contents

Additional README.md files found inside each directory 
with specific directions to run that directory's code. 

- `AuxillaryFunctions`
	* Shared Functions Common to several models, including
	generating labels and training samples from database pickle files
	or evaluation functions used to match detections to GT labels

- `Pickle_Databases`
	* Creates pickle files (using the `pickle` python module)
	to store each of the datasets used by TallyNet and benchmarks
	* Raw files for both the [OREBA](http://www.newcastle.edu.au/oreba)[^OrebaDataPaper] and [Clemson](http://cecas.clemson.edu/~ahoover/cafeteria/)[^ClemsonDataPaper] datasets used can be found at their respective websites.
	
- `Reimp_CTC`
	* Contains reimplementation of [CTC Sequencing Benchmark](https://ieeexplore.ieee.org/document/9305279)[^RouastCTCPaper] 
	* Much of the directory consists of modified GitHub Repositories 
	origianlly published by Phillip Rouast, 
	with some additional files to create results for this paper.
		- Modifications extend the original code to additional dataset variations,
		including the One Handed OREBA Dominant Intake, One Hand OREBA Dominant and Non-Dominant Intake, and Clemson Dominant and Non-Dominant Intake 
	* Orignal Source Repositories can be found at the following links:
		- (CTC Data Processing)[https://github.com/prouast/inertial-sensor-processing]
		- (CTC TensorFlow Custom Layer)[https://github.com/prouast/ctc-beam-search-op]
		- (CTC Intake Detection)[https://github.com/prouast/ctc-intake-detection/]
		
- `Reimp_DongHeuristics`
	* Contains reimplementation of [Roll Heuristic Benchmark](https://link.springer.com/article/10.1007/s10484-012-9194-1)[^DongHeurPaper],
	which uses roll velocity and time thresholds to detect intake gestures.
	
- `Reimp_HeydCNN`
	* Contains reimplmentation of [Binary Classifier CNN-LSTM Benchmark](https://ieeexplore.ieee.org/document/9187203)[^HeydCNNPaper],
	which uses 2 second inputs to predict the probability 
	that the input is an intake gesture, 
	and post-processes these probabilities into detections.
	
- `TallyNet`
	* Proposed novel approach for intake detection,
	which provides a model with long input windows and trains the model
	to predict the total number (the tally) of intakes in the window.
	Decoding these tallies across the meal produce a set of detections 
	for when the events happened.
	* Window size can be set to any value 
	(windows smaller than 5 seconds may require modifications to the 
	CNN kernel sizes),
	with results in this paper spanning 5, 10, 15, 20, and 25 seconds.
	
- `ImprovementsToMake.txt`
	* A file with potential future changes to make to the code.

- `requirements_env1.txt`
	* Full Requirements needed to set up libraries and dependencies
	used in all directories EXCEPT CTC Sequencing Benchmark
- `requirements_env2.txt`
	* Full Requirements needed to set up libraries and dependencies
	SOLELY for CTC Sequencing Benchmark


## Environment Setup

To run code, two environments must be created to run the Python code.

The first environment is detailed in `requirements_env1.txt`,
and contains the dependencies for TallyNet folder,
roll heuristic benchmark, and the binary classifier CNN-LSTM.

The second environment, used exclusively for the CTC sequencing benchmark,
is detailed in `requirements_env2.txt`.
This additional environment is necessitated by 
the custom TensorFlow layer used in the original author's code,
which require specific version of libraries to run without error.
Using more recent library versions with the CTC sequencing benchmark 
may result in errors due to a lack of backward compatible updates
in some libraries.


## Data Prep

Before running TallyNet, Roll Hueristic, or the Binary Classifier CNN-LSTM, 
the raw datasets must be processed into the expected pickle file format
in the `Pickle_Databases` directory.

For the CTC Sequencing Benchamrk, follow instructions inside the data-processing subfolder
to generate TF-Records based on the methods provided by the benchmark's original authors.


## Usage

Once data files have been processed,
training and evaluation can proceed following instuctions provided in each directory.
Remember to activate the proper environment 
(env2 for CTC sequencing, and env1 for all others)
before executing code in a directory.


## Paper Models and Detections

While a pre-trained example model is included for each neural network method,
results in the paper are averages of multiple training runs.
These models and the corresponding detections (provided as a simple list of timesteps)
can be found at [this repository](http://cecas.clemson.edu/~ahoover/).



## References

[^TallyNetPaper]: TBD TallyNet Reference

[^DongHeurPaper]: Y. Dong, A. Hoover, J. Scisco, and E. Muth, 
“A new method for measuring meal intake in humans
via automated wrist motion tracking,”
Applied psychophysiology and biofeedback, vol. 37, pp. 205–215, 2012

[^HeydCNNPaper]: H. Heydarian, P. V. Rouast, M. T. P. Adam, T. Burrows, C. E. Collins and M. E. Rollo,
"Deep Learning for Intake Gesture Detection From Wrist-Worn Inertial Sensors: 
The Effects of Data Preprocessing, Sensor Modalities, and Sensor Positions," 
in IEEE Access, vol. 8, pp. 164936-164949, 2020, doi: 10.1109/ACCESS.2020.3022042.
keywords: {Sensors;Machine learning;Data models;Data preprocessing;Accelerometers;Gyroscopes;Hidden Markov models;Accelerometer;deep learning;intake gesture detection;gyroscope;wrist-worn}

[^RouastCTCPaper]: P. V. Rouast and M. T. P. Adam, 
"Single-Stage Intake Gesture Detection Using CTC Loss
and Extended Prefix Beam Search,"
in IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 7, pp. 2733-2743, July 2021, 
doi: 10.1109/JBHI.2020.3046613.
keywords: {Decoding;Monitoring;Training;Task analysis;Timing;Estimation;Biological system modeling;CTC;deep learning;dietary monitoring;inertial and video sensors;intake gesture detection}

[^OrebaDataPaper]: P. V. Rouast, H. Heydarian, M. T. P. Adam and M. E. Rollo,
"OREBA: A Dataset for Objectively Recognizing Eating Behavior and Associated Intake," 
in IEEE Access, vol. 8, pp. 181955-181963, 2020, doi: 10.1109/ACCESS.2020.3026965.
keywords: {Sensors;Monitoring;Deep learning;Accelerometers;Annotations;Cameras;Synchronization;Dietary monitoring;eating behavior assessment;accelerometer;communal eating;gyroscope;360-degree video camera}

[^ClemsonDataPaper]: Y. Shen, J. Salley, E. Muth and A. Hoover,
"Assessing the Accuracy of a Wrist Motion Tracking Method for Counting Bites Across Demographic and Food Variables," 
in IEEE Journal of Biomedical and Health Informatics, vol. 21, no. 3, pp. 599-606, May 2017, 
doi: 10.1109/JBHI.2016.2612580.
keywords: {Wrist;Tracking;Containers;Sensitivity;Monitoring;Sensors;Correlation;Activity recognition;energy intake;gesture recognition;mHealth}

