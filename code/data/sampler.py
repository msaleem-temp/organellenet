"""
Balanced sampling for class-imbalanced patch datasets.
"""

import sys
import os
import collections
import torch
from torch.utils.data import WeightedRandomSampler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def create_balanced_sampler(dataset) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler that balances class frequencies.

    Each patch is weighted inversely proportional to the frequency of its
    class in the dataset, ensuring rare organelles are sampled more often.

    Parameters
    ----------
    dataset : PatchDataset
        A dataset with a `.patches` attribute (list of dicts with a "class" key).

    Returns
    -------
    WeightedRandomSampler
        A sampler suitable for use with DataLoader.
    """
    class_list = [patch.get("class", "unknown") for patch in dataset.patches]
    class_counts = collections.Counter(class_list)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[patch.get("class", "unknown")] for patch in dataset.patches]

    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )
    return sampler
