# Directory README: Reimp_DongHeuristics

This directory contains all files needed to reproduce results from the Roll Heuristic benchmark method,
which was orignally proposed by Dong *et al.*[^DongHeurPaper].
This method uses a combination of roll velocity and timing thresholds 
to detect when intake occurred during a meal.

This method is only intended to work on data from recordings of the dominant hand
and is not adapted to work on two handed recordings.
Additionally, this method uses the raw data (roll velocity in deg/sec)
rather than Z-Score normalized values often used in machine learning data. 

Before running code, be sure to activate `env1` to load all dependencies.


## Usage

To run the code, type in the command with appropriate values for input arguments.

```
python DongHeur_Benchmarks.py [Dataset] [roll_thresh] [time2_thresh_sec]
```

Results in Paper were tuned to produce balanced scores are provided below.

|Dataset Name          | Dataset | roll_thresh | time2_thresh_sec |
| :---:                | :---:   | :---:       | :---:            |
| DND - One Hand OREBA | 1 | 10 | 8 |
| DND - Clemson        | 2 | 10 | 6 |
| Dom - One Hand OREBA | 3 | 25 | 6 |
| Dom - Clemson        | 4 | 10 | 8 |
<!-- end of table -->



## References

[^DongHeurPaper]: Y. Dong, A. Hoover, J. Scisco, and E. Muth, 
“A new method for measuring meal intake in humans
via automated wrist motion tracking,”
Applied psychophysiology and biofeedback, vol. 37, pp. 205–215, 2012