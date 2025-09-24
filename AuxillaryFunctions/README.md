**NOTE**
GenerateClassDataFuncs.py is actually v2_GenerateClassDataFuncs.  Use and test if it is backward compatible with all directories.

**IDEAS**
- Try to implement a spike detection method: anytime the probability spikes above the current "expected" total, place a detection
    - Must keep a calculate of all prior detections to account for the case where a bite falls out of the window as a new bite slides in

