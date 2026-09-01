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

-> **Status:** 🟢 Done

-> **Result:** 70% crops for training and 15-15 crops for val and testing.



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

**Task:** Filtering Correct Target Class and Encoding them into 13 classes.

-> **Hypotheses:**
1. While filtering target classes, we are using 13 instance classes, During filtering, we should encode 40 classes into 13 main classes. 


-> **Status:** 🔵 In Progress

-> **Result:** N/A

_____________________________________________________________________
**Date:** Sep 1, 2026

**Task:** Classes Encoding in PyTorch dataset Class

-> **Hypotheses:**
1. Currently, in pytorch dataset class, semantic organelle classes such as mito_lum, _mito_mem and mito_rob as encoded as 1:3,1:4,1:5, but mito never encoded whose ID is 50. As result, model as 50 as background. 

-> **Status:** 🟢 Done

-> **Result:** Hypotheses was correct. With previous encoding, model never learned real classes.

_____________________________________________________________________
**Date:** Sep 1, 2026

**Task:** Adjust PyTorch Sampler.

-> **Hypotheses:**
1. Pytorch sampler is based on all classes individually and which distributes weights into classes. It should not do that. Correct logic is it should be based on sum of all classes. 

-> **Status:** 🟢 Done

-> **Result:** N/A

_____________________________________________________________________
**Date:** Sep 1, 2026

**Task:** Update Code.

-> **Status:** 🟡 Pending

