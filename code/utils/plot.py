import matplotlib
import matplotlib.pyplot as plt


def plot_slice(train_dataloader):
    for batch_idx, (em_batch, lbl_batch) in enumerate(train_dataloader):
        print(f"EM Batch Shape:    {em_batch.shape} | Type: {em_batch.dtype}")
        print(f"Label Batch Shape: {lbl_batch.shape} | Type: {lbl_batch.dtype}")

        em_tensor = em_batch[1]
        lbl_tensor = lbl_batch[1]
        print(f"em_tensor {em_tensor.shape}")

        break

    em_single = em_tensor.squeeze(0).numpy()
    lbl_single = lbl_tensor.numpy()

    patch_dim = 128
    mid_z = 64

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')

    axes[0].imshow(em_single[mid_z], cmap='gray')
    axes[0].set_title(f"EM Patch (Slice {mid_z})", color='white')
    axes[0].axis('off')

    # Background maps to 0 now, but keeping your -1 failsafe check
    visual_lbl = np.where(lbl_single[mid_z] == -1, 0, lbl_single[mid_z])
    axes[1].imshow(visual_lbl, cmap='nipy_spectral', interpolation='nearest')
    axes[1].set_title(f"Label Patch (Slice {mid_z})", color='white')
    axes[1].axis('off')

    for ax in axes:
        ax.axhline(mid_z, color='blue', linestyle='-', linewidth=1)
        ax.axvline(mid_z, color='blue', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.show()