import SimpleITK as sitk
import os
import sys

# --- CONFIGURATION ---
# Point to your RAW folder
RAW_FOLDER = r"D:\Dr.Makary\open_ms_data\longitudinal\raw\patient11"
PATH_FIXED = os.path.join(RAW_FOLDER, "study1_FLAIR.nii.gz")
PATH_MOVING = os.path.join(RAW_FOLDER, "study2_FLAIR.nii.gz")

# Output
OUTPUT_PATH = r"D:\MS_Lesion_Tracking\Outputs\patient11\TEST_Robust_Alignedd.nii"

def command_iteration(method):
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f} : {method.GetOptimizerPosition()}")

def robust_registration():
    print("🚀 STARTING ROBUST MULTI-STAGE REGISTRATION...")

    # 1. Load Images
    if not os.path.exists(PATH_FIXED) or not os.path.exists(PATH_MOVING):
        print("❌ Error: Check your paths!")
        return
        
    fixed = sitk.ReadImage(PATH_FIXED, sitk.sitkFloat32)
    moving = sitk.ReadImage(PATH_MOVING, sitk.sitkFloat32)

    # 2. Initialization (Center of Mass)
    # We use MOMENTS (brightness center) instead of GEOMETRY to ensure brains overlap initially
    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    
    print("🧠 Initial alignment (Center of Mass) calculated.")

    # 3. Set up Registration
    R = sitk.ImageRegistrationMethod()
    
    # Similarity Metric
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)

    # Optimizer (The "Smart" part)
    # Learning Rate 4.0, Minimum Step 0.01. It shrinks automatically.
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=0.001, numberOfIterations=200, relaxationFactor=0.5)
    R.SetOptimizerScalesFromPhysicalShift()

    # Initial Transform
    R.SetInitialTransform(tx, inPlace=False)
    
    # Interpolator
    R.SetInterpolator(sitk.sitkLinear)

    # ⚠️ MULTI-RESOLUTION STRATEGY (The Pyramid) ⚠️
    # 3 levels: 
    #  - Level 1: Image shrunk by 4x, smoothed by 2px (Rough fit)
    #  - Level 2: Image shrunk by 2x, smoothed by 1px (Medium fit)
    #  - Level 3: Full size, no smoothing (Perfect fit)
    R.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Monitor progress
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    print("🏃 Execution started... (This may take 30 seconds)")
    try:
        final_tx = R.Execute(fixed, moving)
    except Exception as e:
        print(f"❌ Registration Failed: {e}")
        return

    print(f"✅ DONE! Final Metric: {R.GetMetricValue():.4f}")
    print(f"   Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")

    # 4. Apply Transform
    print("💾 Resampling and saving...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(final_tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    aligned = resampler.Execute(moving)
    sitk.WriteImage(aligned, OUTPUT_PATH)
    
    print(f"✅ Saved to: {OUTPUT_PATH}")
    print("👉 Now run 'math_align_verify.py' on THIS new file!")

if __name__ == "__main__":
    robust_registration()