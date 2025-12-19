import os
import nibabel as nib
import numpy as np

# --- CONFIGURATION ---
mask_folder = r"D:\MS_Lesion_Tracking\Outputs\patient17"

# --- HELPER FUNCTION ---
def get_clean_volume(mask_filename):
    # 1. Load the file
    path = os.path.join(mask_folder, mask_filename)
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path += ".gz"
    
    img = nib.load(path)
    data = img.get_fdata()
    
    # 2. Apply "Digital Eyelid" (Ignore bottom 28 slices)
    data[:, :, :28] = 0
    
    # 3. Count non-zero pixels (Lesions)
    pixel_count = np.sum(data > 0)
    
    # 4. Convert to Volume (mm³)
    # We multiply by the voxel size (usually 1x1x1 mm or similar)
    voxel_dims = img.header.get_zooms()
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
    
    total_volume_mm3 = pixel_count * voxel_volume
    return total_volume_mm3

# --- CALCULATE ---
print("📊 CALCULATING DISEASE PROGRESSION...")

try:
    vol_t1 = get_clean_volume("Case01_S1.nii")
    vol_t2 = get_clean_volume("Case01_S2.nii")

    print(f"\n🔹 Timepoint 1 (Baseline) Volume:  {vol_t1:.2f} mm³")
    print(f"🔹 Timepoint 2 (Follow-up) Volume: {vol_t2:.2f} mm³")

    # Calculate Difference
    diff = vol_t2 - vol_t1
    percent_change = (diff / vol_t1) * 100

    print("\n------------------------------------------------")
    if diff > 0:
        print(f"⚠️ CONCLUSION: DISEASE PROGRESSION DETECTED")
        print(f"   The lesion load INCREASED by +{percent_change:.1f}% (+{diff:.2f} mm³)")
    elif diff < 0:
        print(f"✅ CONCLUSION: TREATMENT RESPONSE DETECTED")
        print(f"   The lesion load DECREASED by {percent_change:.1f}% ({diff:.2f} mm³)")
    else:
        print(f"⚖️ CONCLUSION: STABLE DISEASE (No Change)")
    print("------------------------------------------------")

except Exception as e:
    print(f"❌ Error calculating volume: {e}")





















# import os
# import matplotlib.pyplot as plt
# import nibabel as nib
# import numpy as np
# from matplotlib.colors import ListedColormap

# # --- CONFIGURATION ---
# pred_folder  = r"D:\MS_Lesion_Tracking\Outputs\patient11"
# gt_folder    = r"D:\MS_Lesion_Tracking\Outputs\patient11\Inference_P11"

# # --- 1. LOAD DATA ---
# def load_nii(path):
#     if not os.path.exists(path) and os.path.exists(path + ".gz"):
#         path += ".gz"
#     return nib.load(path).get_fdata()

# gt_path = os.path.join(gt_folder, "gt.nii")
# pred_path = os.path.join(pred_folder, "Case01_S1.nii")
# scan_path = os.path.join(gt_folder, "study1_FLAIR.nii")

# gt_data = load_nii(gt_path)
# pred_data = load_nii(pred_path)
# scan_data = load_nii(scan_path)

# # --- 2. APPLY "DIGITAL EYELID" (CLEANING) ---
# # Create a cleaned copy of the prediction
# clean_pred = pred_data.copy()
# # Force everything below Slice 28 to be 0 (Background)
# clean_pred[:, :, :28] = 0 

# print("🧹 Applied 'Digital Eyelid' (Removed bottom 28 slices of noise).")

# # --- 3. CALCULATE SCORES ---
# def get_dice(truth, prediction):
#     flat_t = truth.flatten()
#     flat_p = prediction.flatten()
#     return (2. * np.sum(flat_t * flat_p)) / (np.sum(flat_t) + np.sum(flat_p))

# raw_score = get_dice(gt_data, pred_data)
# clean_score = get_dice(gt_data, clean_pred)

# print(f"\n📊 RAW Accuracy (with Eyes):   {raw_score:.4f}")
# print(f"🚀 CLEAN Accuracy (Brain only): {clean_score:.4f}")
# print(f"   (Improvement: +{(clean_score - raw_score)*100:.1f}%)")

# # --- 4. FIND THE "BEST" SLICE (Most Green) ---
# # We want to see where the model worked BEST
# overlap_map = (gt_data == 1) & (clean_pred == 1)
# slice_scores = np.sum(overlap_map, axis=(0, 1))
# best_slice = np.argmax(slice_scores)

# if np.max(slice_scores) == 0:
#     print("⚠️ No perfect overlap found. Showing a slice with Ground Truth lesions instead.")
#     best_slice = np.argmax(np.sum(gt_data, axis=(0, 1)))

# # --- 5. VISUALIZE THE SUCCESS ---
# print(f"\n👀 Showing 'Cleaned' Prediction on Slice {best_slice}...")

# fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# # A. Ground Truth
# ax[0].imshow(scan_data[:, :, best_slice], cmap='gray')
# masked_gt = np.ma.masked_where(gt_data[:, :, best_slice] == 0, gt_data[:, :, best_slice])
# ax[0].imshow(masked_gt, cmap='Greens', alpha=0.7)
# ax[0].set_title("Ground Truth")
# ax[0].axis('off')

# # B. Cleaned Prediction
# ax[1].imshow(scan_data[:, :, best_slice], cmap='gray')
# masked_pred = np.ma.masked_where(clean_pred[:, :, best_slice] == 0, clean_pred[:, :, best_slice])
# ax[1].imshow(masked_pred, cmap='Reds', alpha=0.7)
# ax[1].set_title("AI Prediction (Cleaned)")
# ax[1].axis('off')

# # C. Difference Map
# diff_map = np.zeros(gt_data.shape)
# diff_map[(gt_data == 1) & (clean_pred == 1)] = 3  # Green (Match)
# diff_map[(gt_data == 0) & (clean_pred == 1)] = 2  # Red (False Pos)
# diff_map[(gt_data == 1) & (clean_pred == 0)] = 1  # Blue (Missed)

# ax[2].imshow(scan_data[:, :, best_slice], cmap='gray')
# cmap_diff = ListedColormap(['none', 'blue', 'red', 'lime'])
# masked_diff = np.ma.masked_where(diff_map[:, :, best_slice] == 0, diff_map[:, :, best_slice])
# ax[2].imshow(masked_diff, cmap=cmap_diff, alpha=0.8)
# ax[2].set_title("Difference Map\n(Lime=Match, Blue=Missed, Red=Extra)")
# ax[2].axis('off')

# plt.tight_layout()
# plt.show()






















# import os
# import glob
# import matplotlib.pyplot as plt
# import nibabel as nib
# import numpy as np

# # --- CONFIGURATION ---
# pred_folder  = r"D:\MS_Lesion_Tracking\Outputs\patient11"
# gt_folder    = r"D:\MS_Lesion_Tracking\Outputs\patient11\Inference_P11"

# # --- 1. LOAD FILES ---
# def load_nii(path):
#     # Helper to handle .nii vs .nii.gz
#     if not os.path.exists(path) and os.path.exists(path + ".gz"):
#         path += ".gz"
#     return nib.load(path).get_fdata()

# print("Loading files...")
# try:
#     # 1. Ground Truth (Manual Annotation)
#     gt_path = os.path.join(gt_folder, "gt.nii")
#     gt_data = load_nii(gt_path)

#     # 2. Model Prediction (Timepoint 1)
#     pred_path = os.path.join(pred_folder, "Case01_S1.nii")
#     pred_data = load_nii(pred_path)

#     # 3. Original Scan (For background)
#     scan_path = os.path.join(gt_folder, "study1_FLAIR.nii")
#     scan_data = load_nii(scan_path)

#     print("✅ Files Loaded Successfully.")

# except Exception as e:
#     print(f"❌ Error loading files: {e}")
#     exit()

# # --- 2. CALCULATE DICE SCORE ---
# # Flatten arrays to 1D for easy calculation
# gt_flat = gt_data.flatten()
# pred_flat = pred_data.flatten()

# # Intersection (Where both are 1)
# intersection = np.sum(gt_flat * pred_flat)
# dice_score = (2. * intersection) / (np.sum(gt_flat) + np.sum(pred_flat))

# print(f"\n🏆 DICE COEFFICIENT (ACCURACY): {dice_score:.4f}")
# if dice_score < 0.5:
#     print("   ⚠️ Note: Low score is likely due to the 'Eye/Nose' false positives.")

# # --- 3. VISUALIZE DIFFERENCES ---
# # Find slice with most GROUND TRUTH lesions
# slice_sums = np.sum(gt_data, axis=(0, 1))
# target_slice = np.argmax(slice_sums)

# print(f"👀 Visualizing verification on Slice {target_slice}...")

# fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# # A. Ground Truth
# ax[0].imshow(scan_data[:, :, target_slice], cmap='gray')
# masked_gt = np.ma.masked_where(gt_data[:, :, target_slice] == 0, gt_data[:, :, target_slice])
# ax[0].imshow(masked_gt, cmap='Greens', alpha=0.7) # Green = Truth
# ax[0].set_title("Ground Truth (Expert)")
# ax[0].axis('off')

# # B. Model Prediction
# ax[1].imshow(scan_data[:, :, target_slice], cmap='gray')
# masked_pred = np.ma.masked_where(pred_data[:, :, target_slice] == 0, pred_data[:, :, target_slice])
# ax[1].imshow(masked_pred, cmap='Reds', alpha=0.7) # Red = Model
# ax[1].set_title("AI Model Prediction")
# ax[1].axis('off')

# # C. Difference Map (The most important one!)
# # Green = Correct (True Positive)
# # Red   = False Positive (Model detected noise, like eyes)
# # Blue  = False Negative (Model missed a real lesion)

# diff_map = np.zeros(gt_data.shape)
# # True Positive (Overlap)
# diff_map[(gt_data == 1) & (pred_data == 1)] = 3  # Green
# # False Positive (Model only)
# diff_map[(gt_data == 0) & (pred_data == 1)] = 2  # Red
# # False Negative (GT only)
# diff_map[(gt_data == 1) & (pred_data == 0)] = 1  # Blue

# ax[2].imshow(scan_data[:, :, target_slice], cmap='gray')
# from matplotlib.colors import ListedColormap
# # Custom map: Transparent, Blue, Red, Green
# cmap_diff = ListedColormap(['none', 'blue', 'red', 'lime'])
# masked_diff = np.ma.masked_where(diff_map[:, :, target_slice] == 0, diff_map[:, :, target_slice])
# ax[2].imshow(masked_diff, cmap=cmap_diff, alpha=0.8)
# ax[2].set_title("Difference Map\n(Green=Correct, Red=Extra, Blue=Missed)")
# ax[2].axis('off')

# plt.tight_layout()
# plt.show()






















# import os
# import glob
# import matplotlib.pyplot as plt
# import nibabel as nib
# import numpy as np

# # --- CONFIGURATION ---
# mask_folder  = r"D:\MS_Lesion_Tracking\Outputs\patient11"
# input_folder = r"D:\MS_Lesion_Tracking\Outputs\patient11\Inference_P11"

# def find_file(folder, prefix):
#     """Searches for a file starting with 'prefix' (ignores .nii vs .nii.gz)"""
#     # Look for ANY file starting with the name (e.g., study1_FLAIR.nii OR .nii.gz)
#     search_pattern = os.path.join(folder, prefix + "*")
#     matches = glob.glob(search_pattern)
    
#     if not matches:
#         return None
#     # Return the first match found
#     return matches[0]

# def plot_montage(scan_prefix, mask_prefix, title):
#     # 1. Auto-Find the actual files
#     scan_path = find_file(input_folder, scan_prefix)
#     mask_path = find_file(mask_folder, mask_prefix)
    
#     # 2. Check if found
#     if not scan_path:
#         print(f"❌ ERROR: Could not find scan starting with '{scan_prefix}' in {input_folder}")
#         return
#     if not mask_path:
#         print(f"❌ ERROR: Could not find mask starting with '{mask_prefix}' in {mask_folder}")
#         return

#     print(f"✅ Found Scan: {os.path.basename(scan_path)}")
#     print(f"✅ Found Mask: {os.path.basename(mask_path)}")

#     # 3. Load Data
#     try:
#         scan = nib.load(scan_path).get_fdata()
#         mask = nib.load(mask_path).get_fdata()

#         # 4. Setup Montage (5 Slices)
#         total_slices = scan.shape[2]
#         indices = [
#             int(total_slices * 0.35), # Lower
#             int(total_slices * 0.45), # Mid-Low
#             int(total_slices * 0.55), # Center
#             int(total_slices * 0.65), # Mid-High
#             int(total_slices * 0.75), # High
#         ]

#         fig, axes = plt.subplots(1, 5, figsize=(20, 5))
#         fig.suptitle(f"{title}\n({os.path.basename(scan_path)})", fontsize=16)

#         for i, idx in enumerate(indices):
#             # Show Scan
#             axes[i].imshow(scan[:, :, idx], cmap='gray')
            
#             # Show Mask (Red) if lesions exist on this slice
#             if np.sum(mask[:, :, idx]) > 0:
#                 masked = np.ma.masked_where(mask[:, :, idx] == 0, mask[:, :, idx])
#                 axes[i].imshow(masked, cmap='autumn', alpha=0.9)
            
#             axes[i].set_title(f"Slice {idx}")
#             axes[i].axis('off')
        
#         plt.show()

#     except Exception as e:
#         print(f"❌ CRASH: Could not load files. Error: {e}")

# # --- EXECUTE ---
# print("🚀 Starting Auto-Discovery Montage...")

# # We only provide the *start* of the filename. The script finds the rest.
# plot_montage("study1_FLAIR", "Case01_S1", "Baseline (Timepoint 1)")
# plot_montage("study2_FLAIR", "Case01_S2", "Follow-up (Timepoint 2)")





# import os
# import matplotlib.pyplot as plt
# import nibabel as nib
# import numpy as np

# # --- 1. CONFIGURATION ---
# # Path to the MASKS (Predictions)
# mask_folder  = r"D:\MS_Lesion_Tracking\Outputs\patient11"

# # Path to the ORIGINAL SCANS (Inputs)
# input_folder = r"D:\MS_Lesion_Tracking\Outputs\patient11\Inference_P11"

# # --- 2. DEFINE PAIRS (Updated to match your screenshot) ---
# pairs = [
#     {
#         "title": "Timepoint 1 (Baseline)",
#         # Mask found in 'patient11' folder
#         "mask": os.path.join(mask_folder, "Case01_S1.nii"), 
        
#         # Scan found in 'Inference_P11' folder (Updated Name!)
#         "scan": os.path.join(input_folder, "study1_FLAIR.nii") 
#     },
#     {
#         "title": "Timepoint 2 (Follow-up)",
#         # Mask found in 'patient11' folder
#         "mask": os.path.join(mask_folder, "Case01_S2.nii"),
        
#         # Scan found in 'Inference_P11' folder (Updated Name!)
#         "scan": os.path.join(input_folder, "study2_FLAIR.nii")
#     }
# ]

# def visualize_overlay(scan_path, mask_path, title, ax):
#     """Loads images and plots the overlay on the slice with the most lesions."""
    
#     # 1. Handle file extensions (.nii vs .nii.gz)
#     if not os.path.exists(mask_path) and os.path.exists(mask_path + ".gz"):
#         mask_path += ".gz"
    
#     # 2. Check if files exist
#     if not os.path.exists(mask_path):
#         print(f"❌ ERROR: Mask file not found: {mask_path}")
#         ax.text(0.5, 0.5, "Mask Not Found", ha='center', va='center')
#         ax.axis('off')
#         return

#     if not os.path.exists(scan_path):
#         # Try checking for .gz version just in case
#         if os.path.exists(scan_path + ".gz"):
#             scan_path += ".gz"
#         else:
#             print(f"❌ ERROR: Scan file not found: {scan_path}")
#             print(f"   (Please check if '{os.path.basename(scan_path)}' is definitely in the folder)")
#             ax.text(0.5, 0.5, "Scan Not Found", ha='center', va='center')
#             ax.axis('off')
#             return

#     # 3. Load Data
#     try:
#         scan_data = nib.load(scan_path).get_fdata()
#         mask_data = nib.load(mask_path).get_fdata()
        
#         # 4. Find interesting slice (most lesions)
#         slice_sums = np.sum(mask_data, axis=(0, 1))
#         target_slice = np.argmax(slice_sums)
        
#         if np.max(slice_sums) == 0:
#             target_slice = mask_data.shape[2] // 2

#         # 5. Plot
#         # Show Scan (Grayscale)
#         ax.imshow(scan_data[:, :, target_slice], cmap='gray')
        
#         # Show Mask (Red overlay)
#         masked_overlay = np.ma.masked_where(mask_data[:, :, target_slice] == 0, mask_data[:, :, target_slice])
#         ax.imshow(masked_overlay, cmap='autumn', alpha=0.6) 
        
#         ax.set_title(f"{title}\nSlice: {target_slice}")
#         ax.axis('off')
#         print(f"✅ Successfully plotted {title}")

#     except Exception as e:
#         print(f"❌ Error loading files: {e}")
#         ax.text(0.5, 0.5, "Error Loading Data", ha='center')

# # --- RUN PLOTTING ---
# print("🚀 Starting Visualization...")

# # Debug: List files to be 100% sure
# print(f"\n📂 Files found in Inference_P11: {os.listdir(input_folder)}")

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# visualize_overlay(pairs[0]["scan"], pairs[0]["mask"], pairs[0]["title"], axes[0])
# visualize_overlay(pairs[1]["scan"], pairs[1]["mask"], pairs[1]["title"], axes[1])

# plt.tight_layout()
# plt.show()