import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

class_names = [
    "Background", "Mitochondria", "Vesicles", "Endosomes", "Lysosomes", 
    "Lipid Droplets", "Nucleus", "Nuclear Pores", "Microtubules", 
    "Peroxisomes", "Golgi", "ER", "ERES"
]

class SegmentationEvaluator:
    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, preds, targets):

        preds = preds.flatten()
        targets = targets.flatten()

        mask = (targets >= 0) & (targets < self.num_classes)

        hist = torch.bincount(
            self.num_classes * targets[mask] + preds[mask], 
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += hist.cpu()

    def get_metrics(self):
        """Computes per-class IoU, Dice, Precision, and Recall from the global confusion matrix."""
        hist = self.confusion_matrix.numpy()

        tp = np.diag(hist)

        fp = hist.sum(axis=0) - tp

        fn = hist.sum(axis=1) - tp

        epsilon = 1e-6
        
        iou = tp / (tp + fp + fn + epsilon)
        dice = (2 * tp) / ((2 * tp) + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        return iou, dice, precision, recall, hist

evaluator = SegmentationEvaluator(num_classes=len(class_names))

# ----------------------- ----------------------- ----------------------- ----------------------- -----------------------


test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# ----------------------- ----------------------- ----------------------- ----------------------- -----------------------


model.eval()


with torch.no_grad():
    for em_batch, lbl_batch in tqdm(test_loader, desc="Evaluating Test Set"):
        em_batch = em_batch.to(device)
        lbl_batch = lbl_batch.to(device)
        
        outputs = model(em_batch)
        predicted_batch = torch.argmax(outputs, dim=1)
        
        evaluator.update(predicted_batch, lbl_batch)
        

iou, dice, precision, recall, conf_matrix = evaluator.get_metrics()



print("\n" + "="*85)
print(f"{'Class':<18} | {'Dice Score':<12} | {'IoU':<12} | {'Precision':<12} | {'Recall':<12}")
print("="*85)

for idx in range(len(class_names)):
    print(f"{class_names[idx]:<18} | {dice[idx]:<12.4f} | {iou[idx]:<12.4f} | {precision[idx]:<12.4f} | {recall[idx]:<12.4f}")

print("="*85)
print(f"Mean Global Dice:      {np.mean(dice):.4f}")
print(f"Mean Global IoU:       {np.mean(iou):.4f}")
print(f"Mean Global Precision: {np.mean(precision):.4f}")
print(f"Mean Global Recall:    {np.mean(recall):.4f}")

