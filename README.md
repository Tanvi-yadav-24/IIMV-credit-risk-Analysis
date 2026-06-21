# IIMV-credit-risk-Analysis
This project describes a hierarchical, two-stage machine learning pipeline for credit
default prediction. Rather than training a single multi-class classifier, the system
decomposestheproblemintotwobinarydecisions: firstidentifyingwhetheraborrower
willdefault, then classifying confirmeddefaults asmild orsevere. Five models compete
at each stage, selected by PR-AUC. Threshold calibration uses F1-optimal search over
pooled out-of-fold predictions from stratified 5-fold cross-validation, removing test-set
leakage from the calibration step entirely. On a held-out test set of 646 observations,
the pipeline achieves 88.5% overall accuracy and a macro F1 of 0.45. No-default
prediction is strong (F1 = 0.94); performance on the two minority default classes
reflects the structural difficulty of the task under severe class imbalance (22 mild
defaults, 13 severe defaults in the test set).
