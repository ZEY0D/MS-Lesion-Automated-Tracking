import SimpleITK as sitk
import os

# # --- 1. SETUP PATHS (Patient 11) ---
# BASE_FOLDER = r"D:\MS_Lesion_Tracking\Outputs\patient11"
# RAW_FOLDER  = r"D:\MS_Lesion_Tracking\Outputs\patient11\Inference_P11"

# # The RAW Brain Scans (Anatomy) - We use these to calculate the movement
# t1_raw_path = os.path.join(RAW_FOLDER, "study1_FLAIR.nii")
# t2_raw_path = os.path.join(RAW_FOLDER, "study2_FLAIR.nii")

# # The AI Mask (The file we want to fix)
# t2_mask_path = os.path.join(BASE_FOLDER, "Case01_S2.nii")

# # The Output Result (The new, fixed mask)
# output_aligned_path = os.path.join(BASE_FOLDER, "Case01_S2_Aligned.nii")



RAW_DATA_FOLDER = r"D:\Dr.Makary\open_ms_data\longitudinal\raw\patient11"

t1_raw_path = os.path.join(RAW_DATA_FOLDER, "study1_FLAIR.nii") # The Anchor
t2_raw_path = os.path.join(RAW_DATA_FOLDER, "study2_FLAIR.nii") # The Mover

# B. WHERE IS THE AI MASK YOU WANT TO FIX?
# (This is the prediction your model made)
MASK_FOLDER = r"D:\MS_Lesion_Tracking\Outputs\patient11"
t2_mask_path = os.path.join(MASK_FOLDER, "Case01_S2.nii") # The Passenger

# C. WHERE DO YOU WANT THE NEW FIXED FILE?
# We give it a special name so we know it came from the raw test
output_aligned_path = os.path.join(MASK_FOLDER, "TEST_Aligned_From_Raw.nii")






def register_and_align():
    print("🔍 Checking files...")
    # Check if files exist (trying both .nii and .nii.gz)
    files_to_check = [t1_raw_path, t2_raw_path, t2_mask_path]
    for i, path in enumerate(files_to_check):
        if not os.path.exists(path) and os.path.exists(path + ".gz"):
            files_to_check[i] += ".gz"
        elif not os.path.exists(path):
            print(f"❌ CRITICAL ERROR: File not found: {path}")
            return

    # Update paths in case extensions changed
    t1_real, t2_real, mask_real = files_to_check

    print("🔄 Loading images for Rigid Registration...")
    fixed_image = sitk.ReadImage(t1_real, sitk.sitkFloat32)   # Baseline (Target)
    moving_image = sitk.ReadImage(t2_real, sitk.sitkFloat32)  # Follow-up (Source)
    mask_image = sitk.ReadImage(mask_real, sitk.sitkFloat32)   # Mask to fix

    print("🧠 Calculating Alignment (Matching Skulls)...")
    
    # 1. Initialize Transform (Centers the images)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.Euler3DTransform(), 
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # 2. Set up Registration Method (Rigid Body: Rotation + Translation)
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity Metric (Mutual Information is best for medical images)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Optimizer (Gradient Descent)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Setup
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 3. Execute Registration
    final_transform = registration_method.Execute(fixed_image, moving_image)
    
    print(f"✅ Alignment Calculated!")
    print(f"   Final Metric Value: {registration_method.GetMetricValue():.4f}")
    print(f"   Stopping Condition: {registration_method.GetOptimizerStopConditionDescription()}")

    # 4. Apply to Mask
    print("🚀 Applying alignment to the Lesion Mask...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)  # Force mask to match T1 dimensions EXACTLY
    resampler.SetTransform(final_transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # Important: Keep binary (0 or 1), no blurring
    resampler.SetDefaultPixelValue(0)
    
    aligned_mask = resampler.Execute(mask_image)

    # 5. Save
    sitk.WriteImage(aligned_mask, output_aligned_path)
    print(f"\n💾 SUCCESS: Saved aligned mask to:\n   {output_aligned_path}")

if __name__ == "__main__":
    register_and_align()























# import SimpleITK as sitk
# import os

# # --- 1. SETUP PATHS (Copied from your find_files.py output) ---
# # The RAW Brain Scans (Anatomy)
# t1_raw_path = r"D:\MS_Lesion_Tracking\Raw_Data\Longitudinal_Tracking\patient19\study1_FLAIR.nii.gz"
# t2_raw_path = r"D:\MS_Lesion_Tracking\Raw_Data\Longitudinal_Tracking\patient19\study2_FLAIR.nii.gz"

# # The AI Mask (Binary) 
# t2_mask_path = r"D:\MS_Lesion_Tracking\Outputs\Case01_S2.nii.gz" 

# # The Output Result
# output_aligned_path = r"D:\MS_Lesion_Tracking\Outputs\Case01_S2_Aligned.nii.gz"

# def register_and_align():
#     global t2_mask_path 
    
#     print("🔍 Checking files...")
#     if not os.path.exists(t1_raw_path):
#         print(f"❌ MISSING RAW T1: {t1_raw_path}")
#         return
#     if not os.path.exists(t2_raw_path):
#         print(f"❌ MISSING RAW T2: {t2_raw_path}")
#         return
#     if not os.path.exists(t2_mask_path):
#         print(f"❌ MISSING MASK: {t2_mask_path}")
#         return
    
#     print("🔄 Loading images for Rigid Registration...")
#     fixed_image = sitk.ReadImage(t1_raw_path, sitk.sitkFloat32)  # Baseline
#     moving_image = sitk.ReadImage(t2_raw_path, sitk.sitkFloat32) # Follow-up
#     mask_image = sitk.ReadImage(t2_mask_path, sitk.sitkFloat32)   # Mask

#     print("🧠 Calculating Alignment (Matching Skulls)...")
#     # 1. Initialize Transform
#     initial_transform = sitk.CenteredTransformInitializer(
#         fixed_image, 
#         moving_image, 
#         sitk.Euler3DTransform(), 
#         sitk.CenteredTransformInitializerFilter.GEOMETRY
#     )

#     # 2. Set up Registration Method
#     registration_method = sitk.ImageRegistrationMethod()
#     registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
#     registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
#     registration_method.SetMetricSamplingPercentage(0.01)
#     registration_method.SetInterpolator(sitk.sitkLinear)
#     registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
#     registration_method.SetOptimizerScalesFromPhysicalShift()
#     registration_method.SetInitialTransform(initial_transform, inPlace=False)

#     # 3. Execute
#     final_transform = registration_method.Execute(fixed_image, moving_image)
#     print(f"✅ Alignment Calculated! Metric: {registration_method.GetMetricValue():.4f}")

#     # 4. Apply to Mask
#     print("🚀 Applying alignment to the Lesion Mask...")
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetReferenceImage(fixed_image) # Match T1 dimensions
#     resampler.SetTransform(final_transform)
#     resampler.SetInterpolator(sitk.sitkNearestNeighbor) # Keep labels 0 or 1
#     resampler.SetDefaultPixelValue(0)
    
#     aligned_mask = resampler.Execute(mask_image)

#     # 5. Save
#     sitk.WriteImage(aligned_mask, output_aligned_path)
#     print(f"💾 Saved aligned mask to: {output_aligned_path}")
#     print("👉 Now update 'final_project_execution.py' to use this file!")

# if __name__ == "__main__":
#     register_and_align()