import os
import json
import zarr
import numpy as np
import glob
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch.nn as nn
from torch.optim import AdamW
from monai.transforms import Compose, RandFlipd, RandRotate90d
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from monai.networks.nets import UNet
from monai.losses import DiceCELoss

from monai.inferers import SlidingWindowInferer
import collections
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree



# ----------------------------------------------------------------------------
# Path 
main_dir = "/mnt/voxelcell_vol1", # project main folder
dataset_dir = f"{main_dir}/raw_data" # If dataset is sub-folder, then no need to update this path
json_dir = f"{main_dir}/all_jsons"
checkpoints_dir = f"{main_dir}/checkpoints"
outputs_dir = f"{main_dir}/outputs"
# ----------------------------------------------------------------------------



json_path = f"{json_dir}/latest_baseline_centroids"
save_path = f"{json_dir}/targets_only.json" 

target_classes = {'endo', 'ld', 'lyso', 'mito', 'mt', 'np', 'nuc', 'perox', 'ves', 'vim', 'golgi', 'er', 'eres'}
excluded_crop = "crop234"

with open(json_path, 'r') as f:
    blueprint = json.load(f)

train_patches = []
excluded_datasets = set()

for patch in blueprint:
    # 1. Check if the patch belongs to the held-out crop
    if patch.get("crop") == excluded_crop:
        excluded_datasets.add(patch.get("dataset"))
        continue # Skip this patch entirely; it belongs to the test set
        
    # 2. If it is a valid training crop, check if it belongs to a target class
    if patch.get("class") in target_classes:
        train_patches.append(patch)

with open(save_path, 'w') as f:
    json.dump(train_patches, f, indent=4)

print(f"Total original patches: {len(blueprint)}")
print(f"Total training patches extracted: {len(train_patches)}")

# 3. Print the dataset name for the excluded crop
if excluded_datasets:
    print(f"\nExcluded {excluded_crop} belonged to the following dataset(s): {', '.join(excluded_datasets)}")
else:
    print(f"\nWarning: {excluded_crop} was not found in the original JSON. Check your crop naming convention.")




input_json = f"{json_dir}/targets_only.json"
output_dir = f"{json_dir}"

with open(input_json, 'r') as f:
    patches = json.load(f)

random.seed(42)
random.shuffle(patches)

total_patches = len(patches)
train_end = int(total_patches * 0.85)
val_end = int(total_patches * 0.94)

# 4. Slice the list into Train (80%), Val (10%), and Test (10%)
train_patches = patches[:train_end]
val_patches = patches[train_end:val_end]
test_patches = patches[val_end:]


train_path = os.path.join(output_dir, "train.json")
val_path = os.path.join(output_dir, "val.json")
test_path = os.path.join(output_dir, "test.json")

with open(train_path, 'w') as f:
    json.dump(train_patches, f, indent=4)
with open(val_path, 'w') as f:
    json.dump(val_patches, f, indent=4)
with open(test_path, 'w') as f:
    json.dump(test_patches, f, indent=4)


print(f"Total Patches: {total_patches}")
print(f"-> Train Dataset (80%): {len(train_patches)}")
print(f"-> Val Dataset   (10%): {len(val_patches)}")
print(f"-> Test Dataset  (10%): {len(test_patches)}")


def extract_safe(zarr_arr, start_coords, patch_shape, pad_value=0, out_dtype=None):
    arr_shape = zarr_arr.shape
    z_min, z_max = max(0, start_coords[0]), min(arr_shape[0], start_coords[0] + patch_shape[0])
    y_min, y_max = max(0, start_coords[1]), min(arr_shape[1], start_coords[1] + patch_shape[1])
    x_min, x_max = max(0, start_coords[2]), min(arr_shape[2], start_coords[2] + patch_shape[2])

    target_dtype = out_dtype if out_dtype is not None else zarr_arr.dtype
    patch = np.full(patch_shape, fill_value=pad_value, dtype=target_dtype)
    
    pz_min, py_min, px_min = z_min - start_coords[0], y_min - start_coords[1], x_min - start_coords[2]
    
    if z_max > z_min and y_max > y_min and x_max > x_min:
        patch[pz_min:pz_min+(z_max-z_min), py_min:py_min+(y_max-y_min), px_min:px_min+(x_max-x_min)] = \
            zarr_arr[z_min:z_max, y_min:y_max, x_min:x_max]
            
    return patch


def build_zarr_map_modal_direct(data_root):
    zarr_map = {}
    
    # Target exactly two levels deep: data_root / folder / dataset.zarr
    search_pattern = os.path.join(data_root, "*", "*.zarr")
    
    # glob.glob without recursive=True is extremely fast here
    for zarr_path in glob.glob(search_pattern):
        dataset_name = os.path.basename(zarr_path).replace(".zarr", "")
        
        if dataset_name not in zarr_map:
            zarr_map[dataset_name] = []
            
        # Prevent duplicates
        if zarr_path not in zarr_map[dataset_name]:
            zarr_map[dataset_name].append(zarr_path)
            
    return zarr_map

# Execute mapping
class Patches(Dataset):
    
    def __init__(self, json_path, zarr_map, patch_dim=128, max_jitter=32):
        self.patch_dim = patch_dim
        self.max_jitter = max_jitter
        self.zarr_map = zarr_map
        
        with open(json_path, 'r') as f:
            raw_patches = json.load(f)
            
        self.zarr_cache = {}

        semantic_to_instance_map = {
            3: 1, 4: 1, 5: 1,                                       # 1. Mitochondria
            8: 2, 9: 2,                                             # 2. Vesicles
            10: 3, 11: 3,                                           # 3. Endosomes
            12: 4, 13: 4,                                           # 4. Lysosomes
            14: 5, 15: 5,                                           # 5. Lipid Droplets
            20: 6, 21: 6, 24: 6, 25: 6, 26: 6, 27: 6, 28: 6, 29: 6, # 6. Nucleus
            22: 7, 23: 7,                                           # 7. Nuclear Pores
            30: 8, 36: 8,                                           # 8. Microtubules
            47: 9, 48: 9,                                           # 9. Peroxisomes
            6: 10, 7: 10,                                           # 10. Golgi Apparatus
            16: 11, 17: 11,                                         # 11. Endoplasmic Reticulum
            18: 12, 19: 12,                                         # 12. ER Exit Sites
            38: 13,                                                 # 13. Vimentin
        }
        
        self.label_lookup = np.zeros(256, dtype=np.int64)
        for semantic_id, instance_id in semantic_to_instance_map.items():
            self.label_lookup[semantic_id] = instance_id

        self.patches = []
        missing_count = 0
        
        for patch in raw_patches:
            dataset = patch["dataset"]
            crop_id = patch["crop"]
            em_lvl = str(patch["em_scale"])
            lbl_lvl = str(patch["label_scale"])
            
            crop_found = False
            
            if dataset in self.zarr_map:
                for zarr_path in self.zarr_map[dataset]:
                    base_recon_path = os.path.join(zarr_path, "recon-1")
                    em_path = os.path.join(base_recon_path, "em", "fibsem-uint8", em_lvl)
                    label_path = os.path.join(base_recon_path, "labels", "groundtruth", crop_id, "all", lbl_lvl)
                    
                    if os.path.exists(em_path) and os.path.exists(label_path):
                        crop_found = True
                        break
                        
            if crop_found:
                self.patches.append(patch)
            else:
                missing_count += 1
                
        print(f"Dataset initialized. Retained {len(self.patches)} valid patches. Pruned {missing_count} missing patches.")

    def __len__(self):
        return len(self.patches)

    def _get_zarr_handles(self, dataset, crop_id, em_scale, label_scale):
        cache_key = f"{dataset}_{crop_id}_{em_scale}_{label_scale}"
        if cache_key in self.zarr_cache:
            return self.zarr_cache[cache_key]
            
        if dataset not in self.zarr_map:
            raise FileNotFoundError(f"Dataset '{dataset}' was not found in the Kaggle directory scan.")
            
        valid_em_path = None
        valid_label_path = None
        
        for zarr_path in self.zarr_map[dataset]:
            base_recon_path = os.path.join(zarr_path, "recon-1")
            temp_em = os.path.join(base_recon_path, "em", "fibsem-uint8", str(em_scale))
            temp_label = os.path.join(base_recon_path, "labels", "groundtruth", crop_id, "all", str(label_scale))
            
            if os.path.exists(temp_label) and os.path.exists(temp_em):
                valid_em_path = temp_em
                valid_label_path = temp_label
                break
                
        if not valid_em_path:
            raise FileNotFoundError(f"Crop {crop_id} for dataset '{dataset}' could not be found in any available parts.")
            
        em_zarr = zarr.open(valid_em_path, mode='r')
        label_zarr = zarr.open(valid_label_path, mode='r')
        
        self.zarr_cache[cache_key] = (em_zarr, label_zarr)
        return em_zarr, label_zarr

    def __getitem__(self, idx):
        patch = self.patches[idx]
        dataset = patch["dataset"]
        crop_id = patch["crop"]
        
        em_lvl = patch["em_scale"]
        lbl_lvl = patch["label_scale"]
        
        # 1. Load exact centers from your JSON
        l_center = np.array(patch["l_center"], dtype=float)
        e_center = np.array(patch["e_center"], dtype=float)
        e_shape = np.array(patch["e_shape"], dtype=int)
        
        # 2. Fetch Cached Zarr Handles
        em_zarr, label_zarr = self._get_zarr_handles(dataset, crop_id, em_lvl, lbl_lvl)
        
        # 3. Base Mathematical Centering
        base_l_start = np.floor(l_center - (self.patch_dim / 2.0)).astype(int)
        base_e_start = np.floor(e_center - (e_shape / 2.0)).astype(int)
        
        # 4. Generate Spatial Jitter using PyTorch RNG
        jitter_vector = torch.randint(-self.max_jitter, self.max_jitter + 1, (3,)).numpy()
        
        # 5. Dynamic Boundary Clamping (Reading directly from the opened Zarr handle)
        lbl_shape = np.array(label_zarr.shape)
        max_l_start = np.maximum(lbl_shape - self.patch_dim, 0)
        
        clamped_l_start = np.clip(base_l_start + jitter_vector, 0, max_l_start)
        effective_jitter = clamped_l_start - base_l_start
        clamped_e_start = base_e_start + effective_jitter
        
        # 6. Extraction
        lbl_np = extract_safe(label_zarr, clamped_l_start, [self.patch_dim, self.patch_dim, self.patch_dim], pad_value=0, out_dtype=np.int64)
        em_np = extract_safe(em_zarr, clamped_e_start, e_shape.tolist(), pad_value=0)
        
        # 7. Semantic Remapping and Tensor Conversion
        remapped_lbl = self.label_lookup[lbl_np]
            
        em_tensor = torch.from_numpy(em_np.astype(np.float32) / 255.0).unsqueeze(0)
        lbl_tensor = torch.from_numpy(remapped_lbl)
        
        return em_tensor, lbl_tensor




def create_balanced_sampler(dataset):
    class_list = [patch.get("class", "unknown") for patch in dataset.patches]
    class_counts = collections.Counter(class_list)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[patch.get("class", "unknown")] for patch in dataset.patches]
    
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    return sampler

dataset_path = build_zarr_map_modal_direct(dataset_dir)
train_json_path = f"{json_dir}/train.json"
val_json_path = f"{json_dir}/val.json"

train_dataset = Patches(train_path, zarr_map=dataset_path, patch_dim=128, max_jitter=48)
val_dataset = Patches(val_path, zarr_map=dataset_path, patch_dim=128, max_jitter=0) # Static for validation

train_sampler = create_balanced_sampler(train_dataset)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=4, 
    sampler=train_sampler, 
    num_workers=2,   
    pin_memory=True,
    drop_last=True  
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=4, 
    shuffle=False, 
    num_workers=2,   
    pin_memory=True,
    drop_last=False  
)






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Training on device: {device} with {num_gpus} GPUs")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=14,  
    channels=(64, 128, 256, 512, 1024),
    strides=(1, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,
    act="PRELU",
    norm="INSTANCE"
).to(device)

if num_gpus > 1:
    model = nn.DataParallel(model)

manual_weights = torch.tensor([
    0.8419,  # 1  Background
    1.0428,  # 2  Mitochondria
    0.8653,  # 3  Vesicles
    0.6660,  # 4  Endosomes
    1.4594,  # 5  Lysosomes
    0.3458,  # 6  Lipid Droplets
    0.6495,  # 7  Nucleus
    1.6585,  # 8  Nuclear Pores
    0.7208,  # 9  Microtubules
    0.1888,  # 10 Peroxisomes
    0.7276,  # 11 Golgi Apparatus
    1.5408,  # 12 Endoplasmic Reticulum
    1.5923,  # 13 ER Exit Sites
    1.7003  # 14 Vimentin
], dtype=torch.float32).to(device)


# -----------------------------------------------------------------------------
criterion = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,
    weight=manual_weights
)



optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = GradScaler()

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

load_checkpoint_path = f"{checkpoints_dir}/resumde_checkpoindt.pth"
save_checkpoint_path = f"{checkpoints_dir}/resumde_checkpoint.pth"
best_model_path = f"{outputs_dir}/baseline.pth"
epoch_log_file = f"{outputs_dir}/training_log.csv"

start_epoch = 0  
num_epochs = 2   
# CHANGED: Initialize to -1.0 for maximization
best_val_dice = -1.0 
early_stopping_patience = 15
early_stopping_counter = 0

accumulation_steps = 2
print_freq = 200

# NEW: Initialize MONAI metric and discretizers
dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = AsDiscrete(argmax=True, to_onehot=14)
post_label = AsDiscrete(to_onehot=14)


if os.path.exists(load_checkpoint_path):
    print(f"Loading checkpoint from {load_checkpoint_path}...")
    checkpoint = torch.load(load_checkpoint_path, map_location=device)
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # CHANGED: Overwrite best_val_dice instead of best_val_loss
    best_val_dice = checkpoint.get('best_val_dice', -1.0)
    print(f"Resuming training starting at epoch {start_epoch}")
else:
    print(f"Historical checkpoint not found at {load_checkpoint_path}. Starting from scratch.")
    start_epoch = 0
    # CHANGED: Added val_dice to log header
    with open(epoch_log_file, "w") as f:
        f.write("epoch,lr,train_loss,val_loss,val_dice\n")

for epoch in range(start_epoch, num_epochs):
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\n=== Epoch [{epoch+1}/{num_epochs}] | LR: {current_lr:.2e} ===")
    
    # --- PHASE 1: TRAINING ---
    model.train()
    train_epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    
    for step, (em_batch, lbl_batch) in enumerate(train_dataloader):
        em_batch = em_batch.to(device)
        if lbl_batch.dim() == 4:
            lbl_batch = lbl_batch.unsqueeze(1)
        lbl_batch = lbl_batch.to(device, dtype=torch.long)
        
        with autocast():
            outputs = model(em_batch)
            loss = criterion(outputs, lbl_batch)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        train_epoch_loss += (loss.item() * accumulation_steps)
        
        if (step + 1) % print_freq == 0:
            print(f"  [Train] Step {step+1}/{len(train_dataloader)} | Loss: {(loss.item() * accumulation_steps):.4f}")
            
    avg_train_loss = train_epoch_loss / len(train_dataloader)
    
    # --- PHASE 2: VALIDATION ---
    model.eval()
    val_epoch_loss = 0.0
    
    with torch.no_grad():
        for step, (em_batch, lbl_batch) in enumerate(val_dataloader):
            em_batch = em_batch.to(device)
            if lbl_batch.dim() == 4:
                lbl_batch = lbl_batch.unsqueeze(1)
            lbl_batch = lbl_batch.to(device, dtype=torch.long)
            
            with autocast():
                outputs = model(em_batch)
                val_loss = criterion(outputs, lbl_batch)
                
            val_epoch_loss += val_loss.item()
            
            # NEW: Apply discretization for metric computation
            val_outputs = [post_pred(i) for i in outputs]
            val_labels = [post_label(i) for i in lbl_batch]
            
            # NEW: Accumulate metric for the batch
            dice_metric(y_pred=val_outputs, y=val_labels)
            
            if (step + 1) % print_freq == 0:
                print(f"  [Val] Step {step+1}/{len(val_dataloader)} | Loss: {val_loss.item():.4f}")
                
    avg_val_loss = val_epoch_loss / len(val_dataloader)
    
    # NEW: Aggregate the final Dice score and reset the metric
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    # CHANGED: Scheduler now steps based on Dice score
    scheduler.step(mean_dice)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {mean_dice:.4f}")
    
    # CHANGED: Append to log file including mean_dice
    if not os.path.exists(epoch_log_file) and epoch == 0:
        with open(epoch_log_file, "w") as f:
            f.write("epoch,lr,train_loss,val_loss,val_dice\n")
    with open(epoch_log_file, "a") as f:
        f.write(f"{epoch+1},{current_lr:.2e},{avg_train_loss:.4f},{avg_val_loss:.4f},{mean_dice:.4f}\n")
        
    # --- PHASE 3: CHECKPOINTING & EARLY STOPPING ---
    resume_state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_dice': best_val_dice  # CHANGED: Saving the best dice score
    }
    
    torch.save(resume_state, save_checkpoint_path)

    # CHANGED: Early stopping evaluates based on maximizing Dice
    if mean_dice > best_val_dice:
        print(f">>> Validation Dice improved from {best_val_dice:.4f} to {mean_dice:.4f}. Saving best model.")
        best_val_dice = mean_dice
        torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), best_model_path)
        early_stopping_counter = 0 
    else:
        early_stopping_counter += 1
        print(f">>> Validation Dice did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
        
    if early_stopping_counter >= early_stopping_patience:
        print(f"\nEarly stopping triggered. Validation Dice has not improved for {early_stopping_patience} consecutive epochs.")
        break

print("\nSession complete.")



# -----------------------------------------------------------------------------


log_path = f"{outputs_dir}/training_log.csv"

if not os.path.exists(log_path):
    raise FileNotFoundError(f"The specified log file was not found: {log_path}")

# Load the historical training data
df = pd.read_csv(log_path)

# Initialize the plot
plt.figure(figsize=(10, 6), facecolor='white')

# Plot training and validation loss
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='orange', linewidth=2)

# Identify where the learning rate decayed (if applicable in this log)
# We find the index where the LR drops and plot a vertical line at that epoch
lr_changes = df[df['lr'].diff() < 0]
for _, row in lr_changes.iterrows():
    decay_epoch = row['epoch']
    new_lr = row['lr']
    plt.axvline(
        x=decay_epoch, 
        color='gray', 
        linestyle='--', 
        alpha=0.7, 
        label=f'LR Decay (-> {new_lr:.2e})'
    )

# Formatting for clarity and professional presentation
plt.title('UNet Baseline 2: Training & Validation Loss Dynamics', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (DiceCE)', fontsize=12)

# Ensure integer ticks on the x-axis for epochs
plt.xticks(range(0, int(df['epoch'].max()) + 1, max(1, int(df['epoch'].max()) // 10)))

plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, linestyle=':')
plt.tight_layout()

# Render the plot
plt.show()





# -----------------------------------------------------------------------------
test_json = f"{json_dir}/test.json"
best_model_path = f"{outputs_dir}/baseline.pth"


test_dataset = Patches(test_json, zarr_map=dataset_path, patch_dim=128, max_jitter=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 2. Model Initialization & Weight Loading
# ---------------------------------------------------------
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=14,  
    channels=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,
    act="PRELU",
    norm="INSTANCE"
).to(device)

model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()  




def compute_hd95(pred_mask, gt_mask):
    
    pred_coords = np.argwhere(pred_mask)
    gt_coords = np.argwhere(gt_mask)

    # If either mask is empty, HD is theoretically infinite. 
    # Return NaN to exclude it from the mean calculation.
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.nan

    # Build KDTrees for fast spatial querying
    pred_tree = cKDTree(pred_coords)
    gt_tree = cKDTree(gt_coords)

    # Find shortest distance from each point in pred to gt, and vice versa
    dist_pred_to_gt, _ = gt_tree.query(pred_coords)
    dist_gt_to_pred, _ = pred_tree.query(gt_coords)

    # 95th percentile of all directed distances
    hd95_pred_to_gt = np.percentile(dist_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(dist_gt_to_pred, 95)

    return max(hd95_pred_to_gt, hd95_gt_to_pred)

class SegmentationEvaluator:
    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        
        # Track HD95 separately: sum of HDs and count of valid patches per class
        self.hd95_sum = np.zeros(num_classes)
        self.hd95_count = np.zeros(num_classes)

    def update(self, preds, targets):

        preds_flat = preds.flatten()
        targets_flat = targets.flatten()
        
        mask = (targets_flat >= 0) & (targets_flat < self.num_classes)
        
        hist = torch.bincount(
            self.num_classes * targets_flat[mask] + preds_flat[mask], 
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist.cpu()

        # 2. Update Hausdorff Distance (Spatial)
        # Convert to numpy once for SciPy operations
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        batch_size = preds_np.shape[0]

        for b in range(batch_size):
            for c in range(self.num_classes):
                pred_c = (preds_np[b] == c)
                target_c = (targets_np[b] == c)
                
                # Only compute if the class exists in either prediction or ground truth
                if pred_c.any() or target_c.any():
                    hd95_val = compute_hd95(pred_c, target_c)
                    if not np.isnan(hd95_val):
                        self.hd95_sum[c] += hd95_val
                        self.hd95_count[c] += 1

    def get_metrics(self):
        hist = self.confusion_matrix.numpy()
        
        tp = np.diag(hist)
        fp = hist.sum(axis=0) - tp
        fn = hist.sum(axis=1) - tp
        
        epsilon = 1e-6
        
        iou = tp / (tp + fp + fn + epsilon)
        dice = (2 * tp) / ((2 * tp) + fp + fn + epsilon)
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        # F1 is mathematically identical to Dice in this context
        f1_score = np.copy(dice)
        
        # Calculate mean HD95 per class, defaulting to 0 if class never appeared
        hd95 = np.divide(
            self.hd95_sum, 
            self.hd95_count, 
            out=np.zeros_like(self.hd95_sum), 
            where=self.hd95_count != 0
        )
        
        return iou, dice, precision, recall, f1_score, hd95, hist

# --- 1. Initialization ---
evaluator = SegmentationEvaluator(num_classes=14)


# --- 2. Evaluation Loop ---
with torch.no_grad():
    for em_batch, lbl_batch in tqdm(test_loader, desc="Evaluating Test Set"):
        em_batch = em_batch.to(device)
        lbl_batch = lbl_batch.to(device)
        
        outputs = model(em_batch)
        predicted_batch = torch.argmax(outputs, dim=1)
        
        evaluator.update(predicted_batch, lbl_batch)

# --- 3. Extract Metrics ---
iou, dice, precision, recall, f1_score, hd95, conf_matrix = evaluator.get_metrics()

# --- 4. Print Global Results ---
class_names = {
    0: "Background", 
    1: "Mitochondria", 
    2: "Vesicles", 
    3: "Endosomes", 
    4: "Lysosomes", 
    5: "Lipid Droplets", 
    6: "Nucleus", 
    7: "Nuclear Pores", 
    8: "Microtubules", 
    9: "Peroxisomes",
    10: "Golgi", 
    11: "ER", 
    12: "ERES",
    13: "Vim"
}

print("\n" + "="*105)
print(f"{'Class':<15} | {'Dice':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'HD95 (px)':<10}")
print("="*105)

for idx in range(14):
    print(f"{class_names[idx]:<15} | {dice[idx]:<10.4f} | {iou[idx]:<10.4f} | {precision[idx]:<10.4f} | {recall[idx]:<10.4f} | {f1_score[idx]:<10.4f} | {hd95[idx]:<10.4f}")

print("="*105)
# Exclude background (index 0) for mean calculations
print(f"Mean Dice:     {np.mean(dice[1:]):.4f}")
print(f"Mean IoU:      {np.mean(iou[1:]):.4f}")
print(f"Mean Precision:{np.mean(precision[1:]):.4f}")
print(f"Mean Recall:   {np.mean(recall[1:]):.4f}")
print(f"Mean F1 Score: {np.mean(f1_score[1:]):.4f}")
print(f"Mean HD95:     {np.mean(hd95[1:]):.4f}")


# -----------------------------------------------------------------------------

def get_scale_trans(base_path, level="s0"):
    """Reads the exact scale and translation offsets from .zattrs"""
    scale, trans = np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])
    try:
        with open(f"{base_path}/.zattrs", 'r') as f:
            meta = json.load(f)
            multiscales = meta.get("multiscales", [{}])[0]
            for ds in multiscales.get("datasets", []):
                if ds.get("path") == level:
                    for t in ds.get("coordinateTransformations", []):
                        if t.get("type") == "scale": scale = np.array(t["scale"])
                        if t.get("type") == "translation": trans = np.array(t["translation"])
                    return scale, trans
            if "coordinateTransformations" in multiscales:
                for t in multiscales["coordinateTransformations"]:
                    if t.get("type") == "scale": scale = np.array(t["scale"])
                    if t.get("type") == "translation": trans = np.array(t["translation"])
    except Exception as e:
        print(f"Warning reading metadata at {base_path}: {e}")
    return scale, trans

def extract_aligned_volumes(dataset_base, crop_id="crop234"):
    print(f"--- Extracting 3D Volumes for {crop_id} ---")
    
    lbl_base = f"{dataset_base}/recon-1/labels/groundtruth/{crop_id}/all"
    em_base = f"{dataset_base}/recon-1/em/fibsem-uint8"

    # 1. Get Scales and Translations
    scale_lbl, trans_lbl = get_scale_trans(lbl_base, "s0")
    scale_em, trans_em = get_scale_trans(em_base, "s0")

    # 2. Open Zarr Arrays (Metadata only, no RAM used yet)
    lbl_zarr = zarr.open(f"{lbl_base}/s0", mode='r')
    em_zarr = zarr.open(f"{em_base}/s0", mode='r')

    lbl_shape = np.array(lbl_zarr.shape)

    # 3. Calculate Spatial Bounding Box for the EM Sub-volume
    phys_min = trans_lbl
    phys_max = (lbl_shape * scale_lbl) + trans_lbl

    e_min = np.round((phys_min - trans_em) / scale_em).astype(int)
    e_max = np.round((phys_max - trans_em) / scale_em).astype(int)

    print(f"Label Crop Shape: {lbl_shape}")
    print(f"Target EM Bounding Box: Z[{e_min[0]}:{e_max[0]}] Y[{e_min[1]}:{e_max[1]}] X[{e_min[2]}:{e_max[2]}]")

    # 4. Extract Data into RAM
    # Extract the full label array
    lbl_volume = np.array(lbl_zarr[:])
    
    # Extract ONLY the matching bounding box from the massive EM array
    em_volume = np.array(em_zarr[e_min[0]:e_max[0], e_min[1]:e_max[1], e_min[2]:e_max[2]])

    # 5. Fix any 1-voxel rounding disparities between the two arrays
    min_z = min(lbl_volume.shape[0], em_volume.shape[0])
    min_y = min(lbl_volume.shape[1], em_volume.shape[1])
    min_x = min(lbl_volume.shape[2], em_volume.shape[2])

    lbl_volume = lbl_volume[:min_z, :min_y, :min_x]
    em_volume = em_volume[:min_z, :min_y, :min_x]

    print(f"Extraction Complete. Final Aligned Shape: {em_volume.shape}")
    
    return em_volume, lbl_volume

# --- Execution ---
dataset_base = f"{dataset_dir}/jrc_cos7-1a/jrc_cos7-1a.zarr"
em_3d, lbl_3d = extract_aligned_volumes(dataset_base, "crop234")


# -----------------------------------------------------------------------------



print(f"Unique classes before remapping: {np.unique(lbl_3d)}")

# 1. Define your mapping dictionary
semantic_to_instance_map = {
    3: 1, 4: 1, 5: 1,                                       # 1. Mitochondria
    8: 2, 9: 2,                                             # 2. Vesicles
    10: 3, 11: 3,                                           # 3. Endosomes
    12: 4, 13: 4,                                           # 4. Lysosomes
    14: 5, 15: 5,                                           # 5. Lipid Droplets
    20: 6, 21: 6, 24: 6, 25: 6, 26: 6, 27: 6, 28: 6, 29: 6, # 6. Nucleus
    22: 7, 23: 7,                                           # 7. Nuclear Pores
    30: 8, 36: 8,                                           # 8. Microtubules
    47: 9, 48: 9,                                           # 9. Peroxisomes
    6: 10, 7: 10,                                           # 10. Golgi Apparatus
    16: 11, 17: 11,                                         # 11. Endoplasmic Reticulum
    18: 12, 19: 12,                                         # 12. ER Exit Sites
    38: 13,                                                 # 13. Vimentin
}

max_raw_id = max(256, lbl_3d.max() + 1)
label_lookup = np.zeros(max_raw_id, dtype=np.int64)

for semantic_id, instance_id in semantic_to_instance_map.items():
    if semantic_id < max_raw_id:
        label_lookup[semantic_id] = instance_id

# 3. Apply the remapping instantly across the entire 3D volume
lbl_3d_remapped = label_lookup[lbl_3d]

# Verify the remapping worked
present_classes = np.unique(lbl_3d_remapped)


print("Preparing tensors for inference...")

# Normalize EM to [0, 1] - adjust this if your training used Z-score normalization
em_normalized = em_3d.astype(np.float32) / 255.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Add Batch and Channel dimensions: [1, 1, Z, Y, X]
em_tensor = torch.tensor(em_normalized).unsqueeze(0).unsqueeze(0).to(device)
lbl_tensor = torch.tensor(lbl_3d_remapped).to(device)


roi_size = (128, 128, 128)
sw_batch_size = 1  # Decrease to 2 or 1 if you hit CUDA OOM errors
overlap = 0.5      # 50% overlap prevents boundary stitching artifacts

inferer = SlidingWindowInferer(roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap)

print("Executing sliding window inference...")
with torch.no_grad():
    # Outputs raw logits of shape [1, 14, Z, Y, X]
    logits = inferer(inputs=em_tensor, network=model)
    
    # Extract the class with the highest probability
    predicted_batch = torch.argmax(logits, dim=1)

# Convert back to NumPy [Z, Y, X] for visualization
pred_3d = predicted_batch.squeeze(0).cpu().numpy()

z_slice = 70

em_slice = em_3d[z_slice, :, :]
lbl_slice = lbl_3d_remapped[z_slice, :, :]
pred_slice = pred_3d[z_slice, :, :]

# ---------------------------------------------------------
# 2. Build Custom Colormap (Black Background + Bright Organelles)
# ---------------------------------------------------------
# Extract distinct bright colors from tab20 (skipping the dark blues/greys)
base_colors = plt.get_cmap("tab20").colors
foreground_colors = list(base_colors[:13])  

# Index 0 is strictly Black
custom_colors = [(0.0, 0.0, 0.0, 1.0)] + foreground_colors
custom_cmap = mcolors.ListedColormap(custom_colors)

# Because we have exactly 13 colors (0 to 12), vmax must be exactly 12
vmin = 0
vmax = 13 

# ---------------------------------------------------------
# 3. Spatial Prediction Visualization
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Plot 1: EM Image
axes[0].imshow(em_slice, cmap="gray")
axes[0].set_title(f"Test EM Image (Z={z_slice})", fontsize=14)
axes[0].axis("off")

# Plot 2: Ground Truth
im1 = axes[1].imshow(lbl_slice, cmap=custom_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
axes[1].set_title("Ground Truth", fontsize=14)
axes[1].axis("off")

# Plot 3: Model Prediction
im2 = axes[2].imshow(pred_slice, cmap=custom_cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
axes[2].set_title("Model Prediction", fontsize=14)
axes[2].axis("off")

# ---------------------------------------------------------
# 4. Dynamic Legend
# ---------------------------------------------------------
unique_classes_in_crop = np.unique(np.concatenate([lbl_3d_remapped.flatten(), pred_3d.flatten()]))

legend_patches = []
for cid in unique_classes_in_crop:
    cid = int(cid)
    color = custom_colors[cid]
    label_text = class_names.get(cid, f"Class {cid}")
    legend_patches.append(mpatches.Patch(color=color, label=f"{cid}: {label_text}"))

fig.legend(
    handles=legend_patches, 
    loc="lower left", 
    bbox_to_anchor=(0.02, 0.02),
    ncol=7, # Increased to 7 to fit the new class
    title="Classes Present in Full 3D Crop", 
    fontsize=11, 
    title_fontsize=13
)

# Squeeze plots to leave room for the legend at the bottom
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.show()


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
