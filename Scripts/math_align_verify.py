import SimpleITK as sitk
import os
import numpy as np

# --- CONFIGURATION ---
# 1. The Anchor (Baseline Raw)
PATH_FIXED = r"D:\Dr.Makary\open_ms_data\longitudinal\raw\patient11\study1_FLAIR.nii"

# 2. The Problem (Follow-up Raw - Misaligned)
PATH_MOVING_RAW = r"D:\Dr.Makary\open_ms_data\longitudinal\raw\patient11\study2_FLAIR.nii"

# 3. The Solution (The Aligned File you just created)
PATH_MOVING_ALIGNED = r"D:\MS_Lesion_Tracking\Outputs\patient11\TEST_Robust_Alignedd.nii"

def get_metrics(img1, img2, name):
    """Calculates mathematical similarity between two images."""
    
    # 1. Resample img2 to match img1 pixel-for-pixel (Required for math)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img1)
    resampler.SetTransform(sitk.Transform()) # Identity transform (just resampling grid)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    img2_res = resampler.Execute(img2)

    # 2. Correlation (Linear relationship)
    # Range: -1 to 1. Higher is Better.
    corr_filter = sitk.SimilarityIndexImageFilter()
    corr_filter.Execute(img1, img2_res)
    correlation = corr_filter.GetSimilarityIndex()
    
    # 3. Mattes Mutual Information (Statistical overlap)
    # We use a registration object just to calculate the metric
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    # We evaluate the metric at the identity transform (0 movement)
    # because we want to know the score of the images AS THEY ARE right now.
    mi_score = reg.MetricEvaluate(img1, img2_res)

    print(f"\n--- {name} ---")
    print(f"📊 Correlation:       {correlation:.4f} (Closer to 1.0 is better)")
    print(f"📉 Mutual Info Cost:  {mi_score:.4f}   (Lower/More Negative is better)")
    
    return correlation, mi_score

def validate_math():
    print("🧮 STARTING MATHEMATICAL DEBUG...")
    
    # Check files
    if not os.path.exists(PATH_MOVING_ALIGNED):
        print("❌ Error: Could not find your 'TEST_Aligned_Raw_Scan.nii'. Did you run the registration?")
        return

    # Load
    fixed = sitk.ReadImage(PATH_FIXED, sitk.sitkFloat32)
    moving_raw = sitk.ReadImage(PATH_MOVING_RAW, sitk.sitkFloat32)
    moving_aligned = sitk.ReadImage(PATH_MOVING_ALIGNED, sitk.sitkFloat32)

    # Calculate
    c1, m1 = get_metrics(fixed, moving_raw, "BEFORE REGISTRATION (Raw)")
    c2, m2 = get_metrics(fixed, moving_aligned, "AFTER REGISTRATION (Aligned)")

    # Conclusion
    print("\n" + "="*30)
    print("🏆 FINAL VERDICT")
    print("="*30)
    
    improvement = ((c2 - c1) / c1) * 100
    if c2 > c1:
        print(f"✅ SUCCESS: Alignment IMPROVED image similarity by {improvement:.1f}%")
        print("   The algorithm successfully pulled the images closer together.")
    else:
        print(f"⚠️ FAILURE: Alignment made things WORSE.")
        print("   The algorithm drifted away from the target.")

if __name__ == "__main__":
    validate_math()