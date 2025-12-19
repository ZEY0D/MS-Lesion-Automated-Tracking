import os
import shutil
import glob

# --- CONFIGURATION (Based on your successful path check) ---
# 1. Source: Where we pasted the 'train' folder in Phase 1
SOURCE_DIR = r"../Raw_Data/Training_MSLesSeg/train"

# 2. Destination: The new nnUNet folders
DEST_BASE = r"../nnUNet_raw/Dataset501_MSLesion"
IMG_DIR = os.path.join(DEST_BASE, "imagesTr")
LBL_DIR = os.path.join(DEST_BASE, "labelsTr")

# Ensure destination exists
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

print(f"🚀 Starting Data Migration...")
print(f"   From: {os.path.abspath(SOURCE_DIR)}")
print(f"   To:   {os.path.abspath(DEST_BASE)}")

# Get all Patient Folders (P1, P2...)
patient_folders = glob.glob(os.path.join(SOURCE_DIR, "P*"))
print(f"   Found {len(patient_folders)} patient folders. Processing...")

success_count = 0

for p_folder in patient_folders:
    p_id = os.path.basename(p_folder) # e.g., "P1"
    
    # Get Timepoint Folders (T1, T2...)
    timepoint_folders = glob.glob(os.path.join(p_folder, "T*"))
    
    for t_folder in timepoint_folders:
        t_id = os.path.basename(t_folder) # e.g., "T1"
        
        # Define Source Filenames (We know they are .nii.gz now)
        # Pattern usually: P1_T1_FLAIR.nii.gz
        src_flair = os.path.join(t_folder, f"{p_id}_{t_id}_FLAIR.nii.gz")
        src_mask = os.path.join(t_folder, f"{p_id}_{t_id}_MASK.nii.gz")
        
        # Check if files exist
        if os.path.exists(src_flair) and os.path.exists(src_mask):
            # --- THE RENAMING MAGIC ---
            # Unique ID: MS_P1_T1
            unique_name = f"MS_{p_id}_{t_id}"
            
            # Destination Paths
            # IMPORTANT: Images MUST end in _0000.nii.gz for nnUNet
            dst_flair = os.path.join(IMG_DIR, f"{unique_name}_0000.nii.gz")
            dst_mask = os.path.join(LBL_DIR, f"{unique_name}.nii.gz")
            
            # Copy the files
            shutil.copy2(src_flair, dst_flair)
            shutil.copy2(src_mask, dst_mask)
            
            success_count += 1
            if success_count % 10 == 0:
                print(f"   ...moved {success_count} scans")

print("-" * 30)
print(f"✅ MIGRATION COMPLETE!")
print(f"   Successfully formatted {success_count} pairs of images.")
print(f"   Check folder: {IMG_DIR}")