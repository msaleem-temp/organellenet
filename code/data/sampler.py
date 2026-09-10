"""
Balanced sampling for class-imbalanced patch datasets.
"""

import sys
import os
import collections
import torch
import json
from torch.utils.data import WeightedRandomSampler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def organelle_class_name(class_name: str) -> str:
    """
    Collapse raw patch classes to their macro-organelle.
    Uses an explicit mapping to handle irregular CellMap ontology names 
    (e.g., chrom -> nuc, mito_ribo -> mito).
    """
    string_to_macro = {
        'mito': 'mito', 'mito_lum': 'mito', 'mito_mem': 'mito', 'mito_ribo': 'mito',
        'ves': 'ves', 'ves_lum': 'ves', 'ves_mem': 'ves',
        'endo': 'endo', 'endo_lum': 'endo', 'endo_mem': 'endo',
        'lyso': 'lyso', 'lyso_lum': 'lyso', 'lyso_mem': 'lyso',
        'ld': 'ld', 'ld_lum': 'ld', 'ld_mem': 'ld',
        'nuc': 'nuc', 'ne': 'nuc', 'ne_mem': 'nuc', 'ne_lum': 'nuc', 
        'chrom': 'nuc', 'echrom': 'nuc', 'hchrom': 'nuc', 'nhchrom': 'nuc', 'nechrom': 'nuc', 
        'nucpl': 'nuc', 'nucleo': 'nuc',
        'np': 'np', 'np_in': 'np', 'np_out': 'np',
        'mt': 'mt', 'mt_in': 'mt', 'mt_out': 'mt',
        'perox': 'perox', 'perox_lum': 'perox', 'perox_mem': 'perox',
        'golgi': 'golgi', 'golgi_lum': 'golgi', 'golgi_mem': 'golgi',
        'er': 'er', 'er_lum': 'er', 'er_mem': 'er',
        'eres': 'eres', 'eres_lum': 'eres', 'eres_mem': 'eres',
        'vim': 'vim'
    }
    return string_to_macro.get(class_name, class_name)


def create_balanced_sampler(dataset, balance_level: str = "raw") -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that balances class frequencies.

    Parameters
    ----------
    dataset : PatchDataset
        A dataset with a `.patches` attribute (list of dicts with a "class" key).
    balance_level : str
        'raw' to balance sub-compartments, 'organelle' to balance macro-classes.

    Returns
    -------
    WeightedRandomSampler
    """
    if balance_level not in {"raw", "organelle"}:
        raise ValueError(
            f"Unknown sampler balance_level={balance_level!r}; "
            "expected 'raw' or 'organelle'."
        )

    if balance_level == "organelle":
        class_list = [
            organelle_class_name(patch.get("class", "unknown"))
            for patch in dataset.patches
        ]
    else:
        class_list = [patch.get("class", "unknown") for patch in dataset.patches]

    class_counts = collections.Counter(class_list)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items() if count > 0}

    if balance_level == "organelle":
        sample_weights = [
            class_weights[organelle_class_name(patch.get("class", "unknown"))]
            for patch in dataset.patches
        ]
    else:
        sample_weights = [
            class_weights[patch.get("class", "unknown")]
            for patch in dataset.patches
        ]

    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )
    

    if balance_level == "organelle":
        print("\nSampler balanced across Macro-Classes:")
        for cls, count in class_counts.most_common():
            if cls != "unknown":
                print(f"{cls:<10}: {count} patches -> weight: {class_weights[cls]:.6f}")

    return sampler


def create_rfs_sampler(dataset, weights_json_path: str, num_samples: int = 1600) -> WeightedRandomSampler:
    """
    Creates a PyTorch WeightedRandomSampler based on offline Repeat Factor Sampling (RFS) weights.
    
    Parameters
    ----------
    dataset : DynamicCropDataset
        The initialized dataset containing a `.crops` attribute.
    weights_json_path : str
        Path to the JSON file containing the pre-calculated crop RFS weights.
    num_samples : int
        The total number of dynamic patches to extract per epoch. Default is 1600.
        
    Returns
    -------
    WeightedRandomSampler
    """
    # 1. Load the offline RFS weights
    with open(weights_json_path, 'r') as f:
        rfs_weights_dict = json.load(f)
        
    # 2. Map the weights using the Dataset's internal list order
    sample_weights = []
    for crop_meta in dataset.crops:
        crop_id = crop_meta["crop"]
        # Default to 1.0 (baseline) if the crop has no rare targets
        weight = rfs_weights_dict.get(crop_id, 1.0)
        sample_weights.append(weight)
        
    # 3. Convert to PyTorch sampler format
    weights_tensor = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=weights_tensor, 
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler