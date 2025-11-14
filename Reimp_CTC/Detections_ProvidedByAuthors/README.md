# Directory README: Reimp_CTC

This directory contains result files provided by the original authors
from Rouast *et al.*[^RouastCTCPaper], 
which is used as the CTC sequence benchmark method 
in the TallyNet paper[^TallyNetPaper].
This method uses CTC loss neural network and 8 second inputs to produce 
a sequence of class proabilities (intake/non-intake, or bite/drink/non-intake),
and decodes these probabilities wtih an extended beam search to produce detections.

Files are for the specific train/valid/eval split used in the original paper,
rather than the 5-fold validation used in TallyNet paper.
Files inside are logits predicted for a given meal at 8 Hz,
with the right most column being the decoded classification.
A value of 1 is the standard non-intake class,
while 2 signifies an intake gesture.




## References

[^RouastCTCPaper]: P. V. Rouast and M. T. P. Adam, 
"Single-Stage Intake Gesture Detection Using CTC Loss
and Extended Prefix Beam Search,"
in IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 7, pp. 2733-2743, July 2021, 
doi: 10.1109/JBHI.2020.3046613.
keywords: {Decoding;Monitoring;Training;Task analysis;Timing;Estimation;Biological system modeling;CTC;deep learning;dietary monitoring;inertial and video sensors;intake gesture detection}

[^TallyNetPaper]: TBD TallyNet Reference
