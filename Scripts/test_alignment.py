import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Use the ORIGINAL files from the 'coregistered' folder
# (Update these paths if they are different on your machine)
# path_scan1 = r"D:\MS_Lesion_Tracking\Outputs\Case01_S1.nii.gz"
path_scan1 = r"D:\Dr.Makary\open_ms_data\longitudinal\coregistered\patient11\study1_FLAIR.nii.gz"
path_scan2 = r"D:\MS_Lesion_Tracking\Outputs\patient11\TEST_Robust_Alignedd.nii"
# path_scan1 = r"D:\Dr.Makary\open_ms_data\longitudinal\coregistered\patient11\study1_FLAIR.nii.gz"
# path_scan2 = r"D:\Dr.Makary\open_ms_data\longitudinal\coregistered\patient11\study2_FLAIR.nii.gz"
# path_scan2 = r"D:\MS_Lesion_Tracking\Outputs\Case01_S2.nii.gz"
# path_scan2 = r"D:\MS_Lesion_Tracking\Outputs\Case01_S2_Aligned.nii"

def visual_alignment_check():
    print("📂 Loading Scans...")

    # Check files
    if not os.path.exists(path_scan1):
        print(f"❌ Error: Baseline file not found: {path_scan1}")
        return
    if not os.path.exists(path_scan2):
        print(f"❌ Error: Follow-up file not found: {path_scan2}")
        return

    # Load images
    img1 = sitk.ReadImage(path_scan1, sitk.sitkFloat32)
    img2 = sitk.ReadImage(path_scan2, sitk.sitkFloat32)

    # Resample img2 to match img1 exactly (just in case of slight grid diffs)
    # This doesn't "register" them, just ensures they have the same pixel grid for plotting
    img2 = sitk.Resample(img2, img1, sitk.Transform(), sitk.sitkLinear, 0.0, img2.GetPixelID())

    # Create Checkerboard
    print("🏁 Generating Checkerboard View...")
    checkerboard = sitk.CheckerBoard(img1, img2, checkerPattern=[5, 5, 5])

    # Convert to numpy for plotting
    nda = sitk.GetArrayFromImage(checkerboard)
    
    # Select a middle slice
    slice_idx = nda.shape[0] // 2

    # --- PLOT ---
    plt.figure(figsize=(10, 10))
    plt.imshow(nda[slice_idx, :, :], cmap='gray')
    plt.title("Alignment Check: Checkerboard View\n(Look for continuous edges)", fontsize=14)
    plt.axis('off')
    plt.show()

    print("✅ Displaying Check. Look at the SKULL boundary.")
    print("   - Smooth Line? -> ALREADY ALIGNED")
    print("   - Broken/Jagged Line? -> NEEDS ALIGNMENT")

if __name__ == "__main__":
    visual_alignment_check()