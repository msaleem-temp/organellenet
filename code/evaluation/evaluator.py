"""
Segmentation evaluation metrics for OrganelleNet.

Computes per-class Dice, IoU, Precision, Recall, F1, and HD95
from model predictions vs ground truth.
"""

import sys
import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def compute_hd95(pred_mask, gt_mask):
    """
    Compute the 95th percentile Hausdorff Distance between two binary masks.

    Parameters
    ----------
    pred_mask : np.ndarray
        Binary prediction mask.
    gt_mask : np.ndarray
        Binary ground truth mask.

    Returns
    -------
    float
        HD95 value, or np.nan if either mask is empty.
    """
    pred_coords = np.argwhere(pred_mask)
    gt_coords = np.argwhere(gt_mask)

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.nan

    pred_tree = cKDTree(pred_coords)
    gt_tree = cKDTree(gt_coords)

    dist_pred_to_gt, _ = gt_tree.query(pred_coords)
    dist_gt_to_pred, _ = pred_tree.query(gt_coords)

    hd95_pred_to_gt = np.percentile(dist_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(dist_gt_to_pred, 95)

    return max(hd95_pred_to_gt, hd95_gt_to_pred)


class SegmentationEvaluator:
    """
    Accumulates confusion matrix and HD95 statistics across batches,
    then computes per-class segmentation metrics.

    Parameters
    ----------
    num_classes : int
        Total number of classes (including background at index 0).
    """

    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )
        self.hd95_sum = np.zeros(num_classes)
        self.hd95_count = np.zeros(num_classes)

    def update(self, preds, targets):
        """
        Update metrics with a batch of predictions and targets.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted class indices, shape [B, Z, Y, X].
        targets : torch.Tensor
            Ground truth class indices, shape [B, Z, Y, X].
        """
        preds_flat = preds.flatten()
        targets_flat = targets.flatten()

        mask = (targets_flat >= 0) & (targets_flat < self.num_classes)

        hist = torch.bincount(
            self.num_classes * targets_flat[mask] + preds_flat[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += hist.cpu()

        # HD95 computation (spatial metric)
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        batch_size = preds_np.shape[0]

        for b in range(batch_size):
            for c in range(self.num_classes):
                pred_c = preds_np[b] == c
                target_c = targets_np[b] == c

                if pred_c.any() or target_c.any():
                    hd95_val = compute_hd95(pred_c, target_c)
                    if not np.isnan(hd95_val):
                        self.hd95_sum[c] += hd95_val
                        self.hd95_count[c] += 1

    def get_metrics(self):
        """
        Compute per-class metrics from the accumulated confusion matrix.

        Returns
        -------
        dict
            Keys: 'iou', 'dice', 'precision', 'recall', 'f1', 'hd95',
            'confusion_matrix'. Each metric is a numpy array of shape (num_classes,).
        """
        hist = self.confusion_matrix.numpy()

        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp

        epsilon = 1e-6

        iou = tp / (tp + fp + fn + epsilon)
        dice = (2 * tp) / ((2 * tp) + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = np.copy(dice)

        hd95 = np.divide(
            self.hd95_sum,
            self.hd95_count,
            out=np.zeros_like(self.hd95_sum),
            where=self.hd95_count != 0,
        )

        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hd95": hd95,
            "confusion_matrix": hist,
        }


def decode_segmentation_output(output, target_type="labels", sdt_threshold=0.0):
    """Decode label logits or SDT regression channels to class indices."""
    if target_type != "sdt":
        return torch.argmax(output, dim=1)

    foreground_scores = output[:, 1:]
    best_scores, best_classes = torch.max(foreground_scores, dim=1)
    pred = torch.zeros_like(best_classes, dtype=torch.long)
    pred[best_scores > sdt_threshold] = best_classes[best_scores > sdt_threshold] + 1
    return pred


def run_evaluation(
    model, test_loader, num_classes, device, class_names=None,
    target_type="labels", sdt_threshold=0.0
):
    """
    Run evaluation on a test set and print results.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model in eval mode.
    test_loader : DataLoader
        Test data loader.
    num_classes : int
        Number of classes.
    device : torch.device
        Device.
    class_names : dict, optional
        Mapping from class index to name.

    Returns
    -------
    dict
        Metrics dictionary from SegmentationEvaluator.get_metrics().
    """
    evaluator = SegmentationEvaluator(num_classes=num_classes)

    model.eval()
    with torch.no_grad():
        for em_batch, lbl_batch in tqdm(test_loader, desc="Evaluating Test Set"):
            em_batch = em_batch.to(device)
            lbl_batch = lbl_batch.to(device)

            outputs = model(em_batch)
            predicted_batch = decode_segmentation_output(
                outputs, target_type=target_type, sdt_threshold=sdt_threshold
            )

            evaluator.update(predicted_batch, lbl_batch)

    metrics = evaluator.get_metrics()

    # Print results
    if class_names is None:
        class_names = {i: f"Class {i}" for i in range(num_classes)}

    print("\n" + "=" * 105)
    print(
        f"{'Class':<15} | {'Dice':<10} | {'IoU':<10} | {'Precision':<10} | "
        f"{'Recall':<10} | {'F1 Score':<10} | {'HD95 (px)':<10}"
    )
    print("=" * 105)

    for idx in range(num_classes):
        print(
            f"{class_names.get(idx, f'Class {idx}'):<15} | "
            f"{metrics['dice'][idx]:<10.4f} | {metrics['iou'][idx]:<10.4f} | "
            f"{metrics['precision'][idx]:<10.4f} | {metrics['recall'][idx]:<10.4f} | "
            f"{metrics['f1'][idx]:<10.4f} | {metrics['hd95'][idx]:<10.4f}"
        )

    print("=" * 105)
    # Exclude background (index 0) for mean calculations
    print(f"Mean Dice:      {np.mean(metrics['dice'][1:]):.4f}")
    print(f"Mean IoU:       {np.mean(metrics['iou'][1:]):.4f}")
    print(f"Mean Precision: {np.mean(metrics['precision'][1:]):.4f}")
    print(f"Mean Recall:    {np.mean(metrics['recall'][1:]):.4f}")
    print(f"Mean F1 Score:  {np.mean(metrics['f1'][1:]):.4f}")
    print(f"Mean HD95:      {np.mean(metrics['hd95'][1:]):.4f}")

    return metrics



def save_metrics_to_file(metrics, class_names, output_path):
    """Save metrics to a text file."""
    with open(output_path, "w") as f:
        f.write(f"{'Class':<15} | {'Dice':<10} | {'IoU':<10} | {'Precision':<10} | "
                f"{'Recall':<10} | {'F1':<10} | {'HD95':<10}\n")
        f.write("=" * 100 + "\n")

        num_classes = len(metrics["dice"])
        for idx in range(num_classes):
            f.write(
                f"{class_names.get(idx, f'Class {idx}'):<15} | "
                f"{metrics['dice'][idx]:<10.4f} | {metrics['iou'][idx]:<10.4f} | "
                f"{metrics['precision'][idx]:<10.4f} | {metrics['recall'][idx]:<10.4f} | "
                f"{metrics['f1'][idx]:<10.4f} | {metrics['hd95'][idx]:<10.4f}\n"
            )

        f.write("=" * 100 + "\n")
        f.write(f"Mean Dice:      {np.mean(metrics['dice'][1:]):.4f}\n")
        f.write(f"Mean IoU:       {np.mean(metrics['iou'][1:]):.4f}\n")
        f.write(f"Mean Precision: {np.mean(metrics['precision'][1:]):.4f}\n")
        f.write(f"Mean Recall:    {np.mean(metrics['recall'][1:]):.4f}\n")
        f.write(f"Mean F1 Score:  {np.mean(metrics['f1'][1:]):.4f}\n")
        f.write(f"Mean HD95:      {np.mean(metrics['hd95'][1:]):.4f}\n")

    print(f"Metrics saved to {output_path}")
