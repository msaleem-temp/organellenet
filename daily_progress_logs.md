# Research Progress & Debugging Log

This log records the identification, investigation, resolution, and validation
of issues encountered during development of the 3D organelle segmentation pipeline.

---

## August 30, 2026

### 1. Data Distribution Bloat

**Objective:**  
Investigate the unexpectedly large number of patches in
`Dynamic_Baseline_Centroids.json` and verify whether the filtering process
was correctly selecting the targeted foreground classes.

**Hypotheses:**
1. `Dynamic_Baseline_Centroids.json` contained 11,292 patches across all
   47 raw classes, while 8,363 patches were associated with the targeted
   13 classes.
2. The discrepancy indicated a potential bug in the class-filtering logic.

**Investigation / Action:**
- Inspected the centroid filtering pipeline.
- Traced how labels were assigned to the targeted-class collection.
- Identified that background labels were inadvertently being appended to
  the targeted classes.

**Finding:**
- The filtering logic was incorrectly including background samples.

**Resolution:**
- Corrected the filtering logic so that only the intended foreground
  classes are included.

**Status:** 🟢 Resolved


---

### 2. Crop-Level Dataset Split

**Objective:**  
Eliminate potential data leakage caused by splitting sampled patches rather
than separating complete crops/volumes.

**Hypotheses:**
1. Dataset splitting should be performed at the crop level rather than the
   patch level to ensure that patches from the same crop cannot appear in
   different dataset partitions.
2. Rare-class representation should be considered when constructing the
   development split so that rare structures are sufficiently represented
   in training.

**Investigation / Action:**
- Changed the splitting strategy from patch-level splitting to crop-level
  splitting.
- Examined the distribution of rare classes across crops.
- Constructed training, validation, and testing partitions at the crop level.

**Result:**
- The dataset is now divided using crop-level separation.
- The resulting split contains approximately 70% training crops and
  15% validation / 15% testing crops.

**Status:** 🟢 Resolved


---

## August 31, 2026

### 3. Centroid Re-extraction Across All Crops

**Objective:**  
Verify that centroid extraction covered the complete dataset.

**Hypotheses:**
1. The dataset contains 289 crops, while the existing centroid JSON contained
   only 268 crops.
2. The remaining 21 crops were missing from the centroid extraction process.

**Investigation / Action:**
- Compared the list of available dataset crops against the crops represented
  in the centroid JSON.
- Identified the missing crops.
- Re-ran centroid extraction across all 289 crops.

**Result:**
- Centroids were successfully extracted across all 289 crops.
- The resulting centroid dataset contains **13,162 patches**.

**Status:** 🟢 Resolved


---

### 4. Target-Class Filtering and Label Encoding

**Objective:**  
Determine where raw dataset labels should be mapped to the 13 target
segmentation classes.

**Hypothesis:**
- Raw labels could be filtered and encoded into the 13 target classes during
  centroid extraction before being passed to the PyTorch dataset.

**Investigation / Action:**
- Examined the interaction between raw dataset labels, centroid metadata,
  and the PyTorch dataset class.
- Tested whether performing the 13-class encoding during preprocessing
  was appropriate.

**Finding:**
- Encoding the raw labels outside the PyTorch dataset introduced unnecessary
  coupling between preprocessing and the model's class representation.

**Resolution:**
- Kept the raw label information during preprocessing.
- Moved the final target-class encoding into the PyTorch dataset pipeline.

**Status:** 🟢 Resolved


---

## September 1, 2026

### 5. PyTorch Class-Encoding Bug

**Objective:**  
Verify that all target organelle classes were correctly mapped to the
foreground class IDs used by the segmentation model.

**Hypothesis:**
- Several semantic organelle components were explicitly mapped to target
  IDs, but the parent `mito` class (raw ID 50) was not mapped.
- Consequently, raw label ID 50 could be interpreted as background by the
  training pipeline.

**Investigation / Action:**
- Inspected the label mapping implemented in the PyTorch dataset.
- Compared raw annotation IDs against the expected 13-class target mapping.
- Traced the handling of raw label ID 50.

**Finding:**
- The hypothesis was confirmed.
- The `mito` class was not correctly mapped to a foreground target class.

**Impact:**
- The previous pipeline could incorrectly treat this foreground class as
  background, preventing the model from learning the intended class.

**Resolution:**
- Corrected the class encoding within the PyTorch dataset.

**Status:** 🟢 Resolved


---

### 6. PyTorch Sampler Logic

**Objective:**  
Verify that the class-aware sampler was computing sampling probabilities
according to the intended class distribution.

**Hypothesis:**
- The existing sampler calculated/distributed weights independently for
  individual classes rather than deriving the sampling distribution from
  the aggregate target-class distribution.

**Investigation / Action:**
- Reviewed the sampler's weight calculation.
- Traced how class frequencies were converted into sampling weights.
- Modified the sampler to use the aggregate class distribution when
  determining sampling weights.

**Resolution:**
- Updated the sampler logic to follow the intended aggregate class
  distribution.

**Validation:**
- Code-level validation completed.
- Quantitative validation will be performed through the baseline training
  run.

**Status:** 🟢 Implemented — training validation pending


---

### 7. Codebase Update

**Objective:**  
Bring the corrected data-processing, encoding, splitting, and sampling
components together into a consistent baseline pipeline.

**Action:**
- Integrated the above corrections into the current codebase.
- Reviewed the affected components for consistency.
- Prepared the corrected pipeline for baseline training.

**Status:** 🟢 Completed


---

## September 2, 2026

### 8. Corrected Baseline Training

**Objective:**  
Establish a reliable baseline after resolving the identified data,
splitting, label-encoding, and sampling issues.

**Rationale:**
Before introducing additional architectural or sampling changes, the
corrected pipeline needs to be trained and evaluated to establish a
trustworthy reference point.

**Action:**
- Started training the corrected baseline model using the updated pipeline.
- The training configuration is being kept fixed during this run so that
  the resulting performance can serve as a reference for subsequent
  experiments.

**Current Status:** 🟡 Training in progress

**Next Steps:**
1. Complete baseline training.
2. Evaluate on the crop-disjoint validation/test protocol.
3. Report overall and per-class metrics.
4. Identify classes/organelles where the model performs well.
5. Identify failure modes and difficult classes.
6. Analyse whether failures are associated with class rarity, morphology,
   contextual information, patch size, or other factors.
7. Use these observations to formulate the next controlled experiment.

---

# Current Research State

The major data-pipeline and implementation issues identified during the
current debugging cycle have been addressed.

The corrected pipeline is now being used to establish a baseline model.
Further experiments will be designed only after analysing the baseline's
performance and failure modes.

**Current phase:** Baseline establishment 🟡
