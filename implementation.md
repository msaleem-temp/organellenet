# Pipeline Upgrade: Repeat Factor Sampling(RFS) & Dynamic Extraction


**Note:** The overall file structure and execution flow remain same with previous pipeline:
`Splitting` -> `ZARR_Map` -> `PyTorch Dataset` -> `Sampler` -> `DataLoader` -> `Model Initialization` -> `Training Loop`.

However, the core data loading mechanism has shifted from foreground-centric patch sampling to dynamic crop-level extraction method.

## 1. Data Splitting (`splits.py`)
The legacy foreground centroid sampling approach has been completely discarded in favor of crop-level routing. 

* **New Function:** Added `split_crop_handler`.
* **Curated Splits:** The 289 total crops are split into **258 Training, 16 Validation, and 15 Testing**. 
* **Validation/Test Integrity:** The validation and test crops were explicitly hand-picked (not randomly) sampled to guarantee that all 13 macro-classes are physically present in the evaluation sets. Random splitting risks dropping rare organelles entirely from the evaluation metrics.
* **Output Artifacts:** Generates `train_crops.json`, `val_crops.json`, and `test_crops.json`.
* **Data Structure:** These JSON files no longer store centroids. They store pure crop metadata (e.g., `crop`, `dataset`, `em_scale`, `label_scale`). Note: The legacy centroid JSON is currently used solely as a master list to extract these baseline scale/dataset details.

## 2. Zarr Mapping
* **Unchanged:** The `build_zarr_map` function remains exactly the same.

## 3. PyTorch Dataset (`dataset.py`)
This component underwent the most significant architectural change. The legacy `PatchDataset` is kept in the file for backward compatibility, but the active training pipeline now uses the newly added `DynamicCropDataset`.

* **Dynamic Alignment:** Instead of loading pre-calculated centroids, the dataset dynamically reads the OME-Zarr `.zattrs` metadata. It uses physical nanometer scales and translation offsets to mathematically map any coordinate in the label volume to its exact voxel in the global EM volume.
* **Infinite Jitter:** Because coordinate selection is handled via random sampling within the crop bounds, spatial jitter is removed. The old `max_jitter` zero.
* **On-the-Fly Encoding:** The dataset automatically applies the 47-to-13 class encoding map (merging raw sub-compartment IDs down to IDs 0-12) instantly after extraction.

## 4. Sampler (`sampler.py`)
To ensure rare organelles are heavily studied by the network despite the random extraction, we implemented RFS at the crop level.

* **New Function:** Added `create_rfs_sampler()`.
* **Mechanism:** It requires the initialized dataset and a `weights_json_path` pointing to offline-computed RFS weights. These weights are pre-calculated based on the volumetric frequency of rare classes across the entire training set.
* **Execution:** The `WeightedRandomSampler` dictates the training distribution. Crops containing rare classes are assigned massive weights, forcing the PyTorch DataLoader to extract patches from them significantly more often than background-heavy crops. 

## 5. Downstream Pipeline
The DataLoader, model initialization, criterion, optimizer, and training loop logic remain unchanged, operating exactly as they did in the baseline pipeline.
