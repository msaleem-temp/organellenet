🟢🟡🔴🔵
# Research & Debugging Logs
_____________________________________________________________________
**Date:** August 30, 2026

**Task:** Data Distribution Bloat

-> **Hypotheses:**
1. Dynamic_Baseline_Centroids.json has total patches -- 11,292 for all 47 classes and filtered/targeted patches -- 8,363 just for 13 classes.
2. There is a bug in filtering these centroids.

-> **Status:** 🟢 Done

-> **Result:** There was a bug that was appending background too into targeted classes.

_____________________________________________________________________
**Date:** August 30, 2026

**Task:** Crop Level Split

-> **Hypotheses:**
1. Instead of patch level splitting, crop level is compolsary to avoide data leakage problem.
2. For that, as there are 289 total crops, 239 for training and 25-25 for Validation and Training. 
3. Instead of randomly crop selection, categorize them WRT to rare classes and keep that crops in training that have more rare representation. 

-> **Status:** 🟡 Pending

-> **Result:** N/A



_____________________________________________________________________
**Date:** August 31, 2026

**Task:** Centroids Re-extraction

-> **Hypotheses:**
1. There are 289 total crops, while current json has only 268 crops. rest of the crops __ 21__ are missing. 
2. Find that crops and extract centroids from all 289 crops. 

-> **Status:** 🟢 Done

-> **Result:** After extracting patches across all 289 crops, total patches are: **13162**


_____________________________________________________________________
**Date:** August 31, 2026

**Task:** Filtering Correct Target Class 

-> **Hypotheses:**
1. While filtering target classes, we are using 13 instance classes, As a result, target_class.json have only these 13 classes. While, in PyTorch dataset classes, we are using IDs of semantic classes which are useless because we have already excluded them while filtering.  


-> **Status:** 🟢 Done

-> **Result:** I found bug in my code, actually it was corrected by Dr. Samia.
