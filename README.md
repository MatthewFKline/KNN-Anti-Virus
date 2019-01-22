# KNN-Anti-Virus

An attempt to improve my original solution to the antivirus assignment.

Uses a K-Nearest Neighbors Algorithm to predict which binaries are benign versus malicious.

[Statement regarding accuracy]

[Statement providing attribution to web articles / guides used]


===============================================================================

Current status / Notes

Error in prediction from MLDetect - "float() argument must be a string or a number" in line 47, knn.fit
Going to try replacing "MALWARE" vs "SAFE" labels with "1" vs "0" labels


===============================================================================

analyzetargetfile - Template script for scraping opcodes from a binary

formattrainingdata - Collect opcodes from all desired binaries (the production version of analyzetargetfile)

generatemodel.py - Generates a model from MLAVData

machinelearningAV.py - Generates a model from MLAVData

machinelearningmodel.sav - Saved KNN model

MLAVData - Collected opcodes from all binaries

MLDetect.py - Presumably makes predictions based on machinelearningmodel.sav; don't know, because it's broken right now

testpredict - Contains sample data from a binary to run a single prediction against



