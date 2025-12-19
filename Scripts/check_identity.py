import nibabel as nib
import os

# --- PATHS TO CHECK ---
# 1. The Raw Patient 19 Data (The "Body")
raw_t1_path = r"D:\MS_Lesion_Tracking\Raw_Data\Longitudinal_Tracking\patient19\study1_FLAIR.nii.gz"
raw_t2_path = r"D:\MS_Lesion_Tracking\Raw_Data\Longitudinal_Tracking\patient19\study2_FLAIR.nii.gz"

# 2. The AI Masks you are using (The "Clothes")
mask_t1_path = r"D:\MS_Lesion_Tracking\Outputs\Case01_S1.nii.gz"
mask_t2_path = r"D:\MS_Lesion_Tracking\Outputs\Case01_S2.nii.gz"

def check_compatibility():
    print("🕵️ IDENTITY CHECK: Do these masks fit this patient?")
    print("-" * 60)
    
    # Check T1 (Baseline)
    if os.path.exists(raw_t1_path) and os.path.exists(mask_t1_path):
        raw_t1 = nib.load(raw_t1_path).shape
        mask_t1 = nib.load(mask_t1_path).shape
        print(f"🔹 BASELINE (Study 1):")
        print(f"   Patient Head Size: {raw_t1}")
        print(f"   AI Mask Size:      {mask_t1}")
        
        if raw_t1 == mask_t1:
            print("   ✅ MATCH! This mask belongs to this patient.")
        else:
            print("   ❌ MISMATCH! This mask is for a different person.")
    else:
        print("⚠️ File missing for T1 check.")

    print("-" * 60)

    # Check T2 (Follow-up)
    if os.path.exists(raw_t2_path) and os.path.exists(mask_t2_path):
        raw_t2 = nib.load(raw_t2_path).shape
        mask_t2 = nib.load(mask_t2_path).shape
        print(f"🔹 FOLLOW-UP (Study 2):")
        print(f"   Patient Head Size: {raw_t2}")
        print(f"   AI Mask Size:      {mask_t2}")
        
        if raw_t2 == mask_t2:
            print("   ✅ MATCH! This mask belongs to this patient.")
        else:
            print("   ❌ MISMATCH! This mask is for a different person.")
    else:
        print("⚠️ File missing for T2 check.")
    print("-" * 60)

if __name__ == "__main__":
    check_compatibility()