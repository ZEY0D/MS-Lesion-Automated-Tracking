import os
import nibabel as nib
import matplotlib.pyplot as plt

""" - Find the files (fixing the path error).

 -- Unlock the compressed files (.nii.gz).

 --- Load the 3D brain scans into memory as a NumPy array (shape 320, 320, 36)."""


# 1. Define path to a specific patient's data (Adjust path if needed)
# Using patient01 from the longitudinal dataset as an example
base_path = "../Raw_Data/Longitudinal_Tracking/patient01" 
study1_path = os.path.join(base_path, "study1_FLAIR.nii.gz")

# 2. Check if file exists
if os.path.exists(study1_path):
    print(f"✅ Success! Found file: {study1_path}")
    
    # 3. Load the image
    img = nib.load(study1_path)
    data = img.get_fdata()
    
    print(f"   Image Shape: {data.shape}")
    print(f"   (This usually looks like (height, width, depth), e.g., 192, 512, 512)")

    # 4. Quick visual check (Middle Slice)
    mid_slice = data.shape[2] // 2
    plt.imshow(data[:, :, mid_slice], cmap='gray')
    plt.title("Patient 01 - Study 1 FLAIR")
    plt.axis('off')
    plt.show()
    
else:
    print(f"❌ Error: Could not find file at {study1_path}")
    print("Check your folder structure in Step 1.1")