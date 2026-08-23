# OrganelleNet

This repository contains the **baseline model** for the project.

## Installation

First, install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Project Directory

Set `main_dir` in the configuration/code to the path of your main project folder:

```python
main_dir = "replace with your project folder"

dataset_dir = f"{main_dir}/raw_data"
json_dir = f"{main_dir}/all_jsons"
checkpoints_dir = f"{main_dir}/checkpoints"
outputs_dir = f"{main_dir}/outputs"
```

Replace `"replace with your project folder"` with the absolute path to your project directory.

## Required Directory Structure

The project directory must follow this structure:

```text
main_project/
│
├── raw_data/
│   ├── dataset_1/
│   ├── dataset_2/
│   ├── ...
│   └── dataset_22/
│
├── all_jsons/
│   └── all_centroids.json
│
├── checkpoints/
│
└── outputs/
```

### Directories

* **`raw_data/`** — Contains the 22 datasets required by the baseline model.
* **`all_jsons/`** — Contains the JSON metadata files. The required file is `all_centroids.json`.
* **`checkpoints/`** — Used to store model checkpoints.
* **`outputs/`** — Used to store model predictions, evaluation results, metrics, and other generated outputs.

## Configuration

Before running the scripts, make sure that:

1. `main_dir` points to the correct project directory.
2. All 22 datasets are present inside `raw_data/` or if data is in another folder, then path can be replace here.
3. `all_centroids.json` is present inside `all_jsons/`.
4. The required Python packages have been installed from `requirements.txt`.

Once the directory structure and dependencies are set up, the baseline model can be run using the provided scripts.
