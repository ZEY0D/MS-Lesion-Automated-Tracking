import os
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
from lesion_tracker import get_sorted_labels, track_longitudinal

# --- CONFIGURATION ---
predictions_dir = r"../Outputs"
baseline_file = os.path.join(predictions_dir, "Case01_S1.nii")
debug_dir = r"../Debug_Output"
os.makedirs(debug_dir, exist_ok=True)

# Handle file extension safety
if not os.path.exists(baseline_file) and os.path.exists(baseline_file + ".gz"):
    baseline_file += ".gz"

def smart_cut_lesion(data, lesion_id):
    """ Creates a gap in the middle of a specific lesion to simulate splitting. """
    # Get coordinates of ONLY this lesion
    coords = np.argwhere(data == lesion_id)
    if len(coords) == 0: return data
    
    # Find the center X coordinate
    x_coords = coords[:, 0]
    mid_x = int(np.mean(x_coords))
    
    # Create a 2-pixel gap
    # We iterate only through the lesion's pixels to avoid deleting healthy tissue
    pixels_removed = 0
    for (x, y, z) in coords:
        if mid_x - 1 <= x <= mid_x + 1: # Cut a 3-pixel wide gap
            data[x, y, z] = 0
            pixels_removed += 1
            
    print(f"      ✂️ Cut logic removed {pixels_removed} voxels from center X={mid_x}")
    return data

def show_verification_plot(orig_data, mod_data, slice_idx, title):
    """ Visualizes Before vs After side-by-side """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    axes[0].imshow(orig_data[:, :, slice_idx], cmap='gray')
    axes[0].set_title("Original (Baseline)", fontsize=14)
    axes[0].axis('off')
    
    # Modified
    axes[1].imshow(mod_data[:, :, slice_idx], cmap='gray')
    axes[1].set_title(f"Modified ({title})", fontsize=14, color='red')
    axes[1].axis('off')
    
    plt.suptitle(f"Visual Verification: Slice {slice_idx}", fontsize=16)
    plt.show()

def run_debug_session():
    print("\n" + "="*40)
    print("🔧 MS LESION TRACKER - DEBUG SUITE")
    print("="*40)
    
    # 1. Load Baseline
    print("📂 Loading Baseline Scan...")
    img = nib.load(baseline_file)
    data = img.get_fdata().astype(int)
    affine = img.affine
    
    # 2. Get Statistics
    labeled_map, num_lesions, ids, sizes = get_sorted_labels(data)
    print(f"✅ Found {num_lesions} lesions.")
    print("-" * 40)
    print("   ID  |  Size (Voxels) |  Center Z-Slice")
    print("-" * 40)
    
    # Create a quick lookup for center slice (to visualize later)
    lesion_centers = {}
    for lid in ids:
        coords = np.argwhere(labeled_map == lid)
        center_z = int(np.mean(coords[:, 2]))
        lesion_centers[lid] = center_z
        print(f"   {lid:02d}  |  {sizes[lid]:<12}  |  {center_z}")
    print("-" * 40)

    # 3. User Menu
    print("\nSelect Manipulation Mode:")
    print(" [1] Force DISAPPEAR (Remove a lesion)")
    print(" [2] Force ENLARGE   (Grow a lesion >25%)")
    print(" [3] Force SHRINK    (Shrink a lesion)")
    print(" [4] Force SPLIT     (Cut a lesion in half)")
    print(" [5] Force NEW       (Add artifical block)")
    
    choice = input("\n👉 Enter Choice (1-5): ").strip()
    
    target_id = None
    if choice in ['1', '2', '3', '4']:
        try:
            target_id = int(input(f"👉 Enter Target Lesion ID (1-{num_lesions}): "))
            if target_id not in ids: raise ValueError
        except:
            print("❌ Invalid ID.")
            return

    # 4. Apply Manipulation
    mod_data = labeled_map.copy()
    mod_data[mod_data > 0] = 1 # Flatten to binary for easier manipulation
    op_name = "Unknown"
    
    if choice == '1': # Disappear
        op_name = "Disappear"
        print(f"⚡ Deleting Lesion {target_id}...")
        mod_data[labeled_map == target_id] = 0
        
    elif choice == '2': # Enlarge
        op_name = "Enlarge"
        print(f"⚡ Enlarging Lesion {target_id}...")
        mask = (labeled_map == target_id)
        # Dilate heavily (3 iterations) to guarantee >25%
        dilated = binary_dilation(mask, iterations=3) 
        mod_data[dilated] = 1
        
    elif choice == '3': # Shrink
        op_name = "Shrink"
        print(f"⚡ Shrinking Lesion {target_id}...")
        mask = (labeled_map == target_id)
        eroded = binary_erosion(mask, iterations=1)
        mod_data[mask] = 0 # Clear old
        mod_data[eroded] = 1 # Write new
        

    elif choice == '4': # Split
        op_name = "Split"
        print(f"⚡ Splitting Lesion {target_id}...")
        
        # [FIX] Use the labeled_map to find WHERE ID 25 is, 
        # but apply the cut to mod_data (which is binary 1s)
        coords = np.argwhere(labeled_map == target_id)
        
        if len(coords) > 0:
            x_coords = coords[:, 0]
            mid_x = int(np.mean(x_coords))
            
            # Cut a wider gap (4 pixels) to ensure the tracker sees 2 separate objects
            # Connectivity=2 (diagonals) can bridge small gaps!
            pixels_removed = 0
            for (x, y, z) in coords:
                if mid_x - 2 <= x <= mid_x + 2: # Expanded from 1 to 2
                    mod_data[x, y, z] = 0
                    pixels_removed += 1
            print(f"      ✂️ Cut logic removed {pixels_removed} voxels.")
        else:
            print("      ⚠️ Target ID not found in map.")


    # elif choice == '4': # Split
    #     op_name = "Split"
    #     print(f"⚡ Splitting Lesion {target_id}...")
    #     mod_data = smart_cut_lesion(mod_data, target_id)

    elif choice == '5': # New
        op_name = "New_Artificial"
        print(f"⚡ creating New Lesion at random empty spot...")
        # Check a safe corner (10,10,10)
        mod_data[10:20, 10:20, 10:20] = 1
        # Set target ID to None because we don't have a "before" ID
        target_id = 999 
        lesion_centers[999] = 15 # Approx slice for visualization

    # 5. Save Debug File
    timestamp = int(time.time())
    save_path = os.path.join(debug_dir, f"Debug_{op_name}_{timestamp}.nii.gz")
    nib.save(nib.Nifti1Image(mod_data.astype(np.float32), affine), save_path)
    print(f"\n💾 Saved modified scan to: {save_path}")

    # 6. Run The TRACKER
    print("\n🚀 Running Logic Engine...")
    _, _, report, _ = track_longitudinal(baseline_file, save_path, silence=True)
    
    # 7. Report Results
    print("\n" + "="*40)
    print("📊 DIAGNOSTIC REPORT")
    print("="*40)
    
    # Filter details for the target we care about
    relevant_details = [d for d in report['Details'] if d['T2_ID'] == target_id or d.get('Size_T1_Ref') != 'N/A']
    
    print(f"Logic Detected: {report['New']} New | {report['Disappeared']} Gone | {report['Enlarged_GT25']} Enlarged | {report['Split_Fragments']} Splits")
    
    for item in report['Details']:
        # If we targeted a specific ID, highlight it. 
        # Note: If we split ID 4, it might become ID 4 and ID 5 in the new list.
        status = item['Status']
        sz = item['Size_T2']
        print(f" > Lesion {item['T2_ID']}: {status} (Size: {sz})")

    # 8. Visual Verification
    print("\n🖼️ Launching Visual Verifier...")
    slice_to_show = lesion_centers.get(target_id, 15)
    # Re-load the saved file to be sure we show exactly what the tracker saw
    loaded_mod = nib.load(save_path).get_fdata()
    
    show_verification_plot(labeled_map, loaded_mod, slice_to_show, op_name)

if __name__ == "__main__":
    run_debug_session()