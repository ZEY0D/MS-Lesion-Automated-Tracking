import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from lesion_tracker import track_longitudinal

# --- CONFIGURATION ---
predictions_dir = r"../Outputs"
file_t1 = os.path.join(predictions_dir, "Case01_S1.nii")
file_t2 = os.path.join(predictions_dir, "Case01_S2.nii")
# file_t2 = os.path.join(predictions_dir, "Case01_S2_Aligned.nii")


# Handle extensions
if not os.path.exists(file_t1) and os.path.exists(file_t1 + ".gz"):
    file_t1 += ".gz"
if not os.path.exists(file_t2) and os.path.exists(file_t2 + ".gz"):
    file_t2 += ".gz"





# that's an important function because we cannot track the lesion progress unless the two images are identical
def resample_to_match(target_img_path, reference_img_path):
    ref_img = nib.load(reference_img_path)
    tgt_img = nib.load(target_img_path)
    ref_data = ref_img.get_fdata()
    tgt_data = tgt_img.get_fdata()
    
    # .shape returns (X x Y x Z)
    if ref_data.shape == tgt_data.shape:
        return tgt_img.get_fdata(), tgt_img.affine
    print(f"⚠️ Resizing Follow-up from {tgt_data.shape} to match Baseline...")
    # here we calculate the scaling factor 
    factors = [r / t for r, t in zip(ref_data.shape, tgt_data.shape)]
    # It ensures values remain strictly integers (0 or 1) avoiding the in between values 
    # thus we use Nearest Neighbor Interpolation
    resized_data = zoom(tgt_data, factors, order=0)
    return resized_data, ref_img.affine









def draw_smart_labels(ax, slice_f, new_ids, img_h, img_w):
    """
    Draws arrows pointing to lesions, pushing text outwards to clear the view.
    """
    unique_ids = np.unique(slice_f)
    center_y, center_x = img_h / 2, img_w / 2
    
    for lesion_id in unique_ids:
        if lesion_id == 0: continue
        
        # Get Lesion Center
        y, x = np.where(slice_f == lesion_id)
        cy, cx = np.mean(y), np.mean(x)
        
        # --- SMART OFFSET LOGIC ---
        # Calculate vector from image center to lesion
        vec_x = cx - center_x
        vec_y = cy - center_y
        
        # Normalize and push out by 25 pixels
        norm = np.sqrt(vec_x**2 + vec_y**2)
        if norm == 0: norm = 1
        
        offset_x = (vec_x / norm) * 25 
        offset_y = (vec_y / norm) * 25
        
        # Text Position (Pushed away from center)
        text_x = cx + offset_x
        text_y = cy + offset_y
        
        # Color Logic
        is_new = lesion_id in new_ids
        l_color = '#FF3333' if is_new else '#00FFFF' # Bright Red vs Cyan
        font_w = 'bold' if is_new else 'normal'
        
        ax.annotate(str(int(lesion_id)), 
                    xy=(cx, cy), xycoords='data',
                    xytext=(text_x, text_y), textcoords='data',
                    color='white', fontsize=7, weight=font_w,
                    arrowprops=dict(arrowstyle="-", color=l_color, lw=0.5),
                    bbox=dict(boxstyle="square,pad=0.1", fc=l_color, alpha=0.5, lw=0))

def visualize_smart_montage(data_f, new_ids):
    """ Shows 9 active slices in a grid. """
    lesion_counts = np.sum(data_f > 0, axis=(0, 1))
    active_slices = np.argsort(lesion_counts)[::-1][:9]
    active_slices = np.sort(active_slices)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Top 9 Active Slices (Overview)\nRed = New | Blue = Stable", fontsize=16)
    
    ax_flat = axes.flatten()
    
    for i, slice_idx in enumerate(active_slices):
        ax = ax_flat[i]
        h, w = data_f.shape[:2]
        rgb_map = np.zeros((h, w, 3))
        slice_f = data_f[:, :, slice_idx]
        
        unique_ids = np.unique(slice_f)
        for lesion_id in unique_ids:
            if lesion_id == 0: continue
            mask = (slice_f == lesion_id)
            if lesion_id in new_ids:
                rgb_map[mask] = [1, 0, 0]
            else:
                rgb_map[mask] = [0, 0, 1]
        
        ax.imshow(rgb_map)
        ax.set_title(f"Slice {slice_idx}", fontsize=10, color='white', backgroundcolor='black')
        ax.axis('off')
        
        draw_smart_labels(ax, slice_f, new_ids, h, w)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_single_slice_zoom(data_f, new_ids, slice_idx):
    """ Shows ONE massive slice for detailed inspection. """
    plt.figure(figsize=(10, 10))
    
    h, w = data_f.shape[:2]
    rgb_map = np.zeros((h, w, 3))
    slice_f = data_f[:, :, slice_idx]
    
    unique_ids = np.unique(slice_f)
    for lesion_id in unique_ids:
        if lesion_id == 0: continue
        mask = (slice_f == lesion_id)
        if lesion_id in new_ids:
            rgb_map[mask] = [1, 0, 0]
        else:
            rgb_map[mask] = [0, 0, 1]
            
    plt.imshow(rgb_map)
    plt.title(f"ZOOM VIEW: Slice {slice_idx}\n(Close this window to choose another)", fontsize=16)
    plt.axis('off')
    
    # Use larger fonts for zoom view
    draw_smart_labels(plt.gca(), slice_f, new_ids, h, w)
    
    plt.show()



def interactive_lesion_inspector(labeled_b, labeled_f, report, t1_data, t2_data):
    """
    Terminal Menu to filter and inspect specific lesion changes.
    Shows T1 (Baseline) vs T2 (Follow-up) side-by-side.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import center_of_mass

    while True:
        print("\n" + "="*50)
        print("🕵️  INTERACTIVE LESION INSPECTOR")
        print("="*50)
        print(f" [1] Show NEW Lesions       ({report['New']} found)")
        print(f" [2] Show DISAPPEARED       ({report['Disappeared']} found)")
        print(f" [3] Show SPLIT Events      ({report['Split_Fragments']} fragments)")
        print(f" [4] Show ENLARGED (>25%)   ({report['Enlarged_GT25']} found)")
        print(f" [5] Show SHRUNK            ({report['Shrunk']} found)")
        print(f" [6] Lookup by ID")
        print(" [Q] Quit & Save Reports")
        
        choice = input("\n👉 Select Category: ").strip().upper()
        
        if choice == 'Q': break
        
        # --- Filter Logic ---
        target_list = []
        if choice == '1':
            target_list = [d for d in report['Details'] if d['Status'] == 'New']
        elif choice == '2':
            target_list = [d for d in report['Details'] if d['Status'] == 'Disappeared']
        elif choice == '3':
            target_list = [d for d in report['Details'] if 'Split' in d['Status']]
        elif choice == '4':
            target_list = [d for d in report['Details'] if d['Status'] == 'Enlarged']
        elif choice == '5':
            target_list = [d for d in report['Details'] if d['Status'] == 'Shrunk']
        elif choice == '6':
            # Just let them type any ID, we'll search for it
            pass
        else:
            print("❌ Invalid selection.")
            continue

        # --- Display Options ---
        if choice != '6':
            if not target_list:
                print("   (None found in this category)")
                continue
            
            print(f"\n📄 Available IDs in this category:")
            for item in target_list:
                # Show T2_ID for existing stuff, or Tracking_ID for disappeared stuff
                disp_id = item['T2_ID'] if item['T2_ID'] != 'N/A' else item['Tracking_ID']
                print(f"   > ID {disp_id} \t| Status: {item['Status']}")
        
        # --- User Selection ---
        try:
            sel_input = input("\n🔍 Enter ID to Inspect: ").strip()
            if not sel_input: continue
            selected_id = int(sel_input)
        except ValueError:
            print("❌ Please enter a number.")
            continue

        # --- Locate the Lesion (Slice Calculation) ---
        # We need to find the best Z-slice to show.
        # If it's Disappeared, it's only in Baseline (labeled_b).
        # If it's New/Stable/Split, it's in Follow-up (labeled_f).
        
        slice_idx = -1
        status_msg = "Unknown"
        
        # Check Follow-up First
        coords_f = np.argwhere(labeled_f == selected_id)
        if len(coords_f) > 0:
            slice_idx = int(np.mean(coords_f[:, 2])) # Center Z
            status_msg = "Present in T2"
        else:
            # Check Baseline (maybe it disappeared?)
            coords_b = np.argwhere(labeled_b == selected_id)
            if len(coords_b) > 0:
                slice_idx = int(np.mean(coords_b[:, 2]))
                status_msg = "Only in T1 (Disappeared)"
        
        if slice_idx == -1:
            print(f"❌ ID {selected_id} not found in either mask.")
            continue

        # --- VISUALIZATION ---
        print(f"   🚀 Opening Comparison View at Slice {slice_idx}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Left: Baseline (T1)
        axes[0].imshow(t1_data[:, :, slice_idx], cmap='gray')
        # Overlay mask if ID exists here
        mask_b = (labeled_b[:, :, slice_idx] == selected_id)
        if np.any(mask_b):
            axes[0].contour(mask_b, colors='blue', linewidths=2)
            axes[0].imshow(np.ma.masked_where(~mask_b, mask_b), cmap='Blues', alpha=0.5, vmin=0, vmax=1)
        axes[0].set_title(f"BEFORE (Baseline)\nSlice {slice_idx}", fontsize=12)
        axes[0].axis('off')

        # Right: Follow-up (T2)
        axes[1].imshow(t2_data[:, :, slice_idx], cmap='gray')
        # Overlay mask if ID exists here
        mask_f = (labeled_f[:, :, slice_idx] == selected_id)
        if np.any(mask_f):
            # Color logic: Red if New, Cyan if Stable/Split
            is_new = any(d['T2_ID'] == selected_id and d['Status'] == 'New' for d in report['Details'])
            cmap_col = 'Reds' if is_new else 'cool'
            cont_col = 'red' if is_new else 'cyan'
            
            axes[1].contour(mask_f, colors=cont_col, linewidths=2)
            axes[1].imshow(np.ma.masked_where(~mask_f, mask_f), cmap=cmap_col, alpha=0.5, vmin=0, vmax=1)
        
        axes[1].set_title(f"AFTER (Follow-up)\nSlice {slice_idx} | ID {selected_id}", fontsize=12)
        axes[1].axis('off')

        plt.suptitle(f"Lesion Inspection: ID {selected_id}\n({status_msg})", fontsize=16)
        plt.tight_layout()
        plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if os.path.exists(file_t1) and os.path.exists(file_t2):
        print(f"📂 Analyzing...")
        
        # 1. Match Dimensions
        fixed_t2_data, affine_ref = resample_to_match(file_t2, file_t1)
        
        # 2. Run Logic Engine
        print("🧠 Running Longitudinal Logic...")
        # Create Nifti object for T2 to pass to tracker
        img_f_obj = nib.Nifti1Image(fixed_t2_data, affine_ref)
        
        labeled_b, labeled_f, report, _ = track_longitudinal(file_t1, img_f_obj)
        
        # Load raw background data for visualization
        raw_t1 = nib.load(file_t1).get_fdata()
        raw_t2 = fixed_t2_data 
        
        print("\n" + "="*50)
        print(f"   Final Count: {report['Followup_Count']} Lesions")
        print(f"   New: {report['New']} | Gone: {report['Disappeared']}")
        print(f"   Events: {report['Merged_Events']} Merges | {report['Split_Fragments']} Splits")
        print("="*50)

        # 3. Show Overview Montage
        print("🖼️ Opening Overview Montage...")
        visualize_smart_montage(labeled_f, report['New_IDs'])
        
        # 4. Interactive Inspector (Terminal Menu)
        interactive_lesion_inspector(labeled_b, labeled_f, report, raw_t1, raw_t2)

        # --- REPORTING ENGINE ---
        print("\n📝 Generating Comprehensive Reports...")
        
        # Summary Report
        summary_data = {k: v for k, v in report.items() if k not in ['Details', 'New_IDs']}
        pd.DataFrame([summary_data]).to_csv(os.path.join(predictions_dir, "Summary_Report.csv"), index=False)
        
        # Detailed Log
        df_details = pd.DataFrame(report['Details'])
        cols = ["Tracking_ID", "Status", "Size_T1_Ref", "Size_T2", "Original_T2_ID", "T2_ID"]
        # Ensure columns exist
        for c in cols:
            if c not in df_details.columns: df_details[c] = "N/A"
            
        df_details = df_details.reindex(columns=cols)
        
        # Sort by Tracking ID
        df_details['Sort_Key'] = pd.to_numeric(df_details['Tracking_ID'], errors='coerce').fillna(9999)
        df_details = df_details.sort_values('Sort_Key').drop(columns=['Sort_Key'])
        
        df_details.to_csv(os.path.join(predictions_dir, "Detailed_Lesion_Log.csv"), index=False)
        print(f"   ✅ Saved: Detailed_Lesion_Log.csv")
        print("   ✅ Saved: Summary_Report.csv")
        
        print("\n👋 Analysis Complete.")
    else:
        print("❌ Error: Files not found.")









#     pd.DataFrame([report]).to_csv(os.path.join(predictions_dir, "Final_Report.csv"), index=False)
    
# else:
#     print("❌ Error: Files not found.")




















# import os
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom
# from lesion_tracker import track_longitudinal

# # --- CONFIGURATION ---
# predictions_dir = r"../Outputs"
# file_t1 = os.path.join(predictions_dir, "Case01_S1.nii")
# file_t2 = os.path.join(predictions_dir, "Case01_S2.nii")

# # Handle extensions
# if not os.path.exists(file_t1) and os.path.exists(file_t1 + ".gz"):
#     file_t1 += ".gz"
# if not os.path.exists(file_t2) and os.path.exists(file_t2 + ".gz"):
#     file_t2 += ".gz"

# def resample_to_match(target_img_path, reference_img_path):
#     ref_img = nib.load(reference_img_path)
#     tgt_img = nib.load(target_img_path)
#     ref_data = ref_img.get_fdata()
#     tgt_data = tgt_img.get_fdata()
    
#     if ref_data.shape == tgt_data.shape:
#         return tgt_img.get_fdata(), tgt_img.affine
    
#     print(f"⚠️ Resizing Follow-up from {tgt_data.shape} to match Baseline...")
#     factors = [r / t for r, t in zip(ref_data.shape, tgt_data.shape)]
#     resized_data = zoom(tgt_data, factors, order=0)
#     return resized_data, ref_img.affine

# def visualize_smart_montage(data_f, new_ids):
#     """
#     Generates a CLEAN 3x3 Montage of the most active slices.
#     Uses arrows for labels to avoid clutter.
#     """
#     # 1. Find the 9 slices with the most lesion pixels
#     lesion_counts = np.sum(data_f > 0, axis=(0, 1))
#     # Get indices of top 9 active slices, sorted by height
#     active_slices = np.argsort(lesion_counts)[::-1][:9]
#     active_slices = np.sort(active_slices) # Sort by Z-index for logical flow

#     fig, axes = plt.subplots(3, 3, figsize=(15, 15))
#     fig.suptitle(f"Top 9 Active Slices (Clean View)\nRed = New | Blue = Stable", fontsize=16)
    
#     ax_flat = axes.flatten()
    
#     for i, slice_idx in enumerate(active_slices):
#         ax = ax_flat[i]
        
#         # Create RGB Map
#         h, w = data_f.shape[:2]
#         rgb_map = np.zeros((h, w, 3))
#         slice_f = data_f[:, :, slice_idx]
        
#         # Color Logic
#         unique_ids = np.unique(slice_f)
#         for lesion_id in unique_ids:
#             if lesion_id == 0: continue
#             mask = (slice_f == lesion_id)
#             if lesion_id in new_ids:
#                 rgb_map[mask] = [1, 0, 0] # RED
#             else:
#                 rgb_map[mask] = [0, 0, 1] # BLUE
        
#         ax.imshow(rgb_map)
#         ax.set_title(f"Slice {slice_idx}", fontsize=10, weight='bold', color='white', backgroundcolor='black')
#         ax.axis('off')
        
#         # --- SMART LABELS WITH ARROWS ---
#         for lesion_id in unique_ids:
#             if lesion_id == 0: continue
            
#             y, x = np.where(slice_f == lesion_id)
#             cy, cx = np.mean(y), np.mean(x)
            
#             # Label Color
#             l_color = 'red' if lesion_id in new_ids else 'cyan'
            
#             # Draw Arrow pointing to the lesion
#             # xy is the target (lesion center), xytext is where the number sits
#             # We offset the text slightly to the right and up
#             ax.annotate(str(int(lesion_id)), 
#                         xy=(cx, cy), xycoords='data',
#                         xytext=(cx + 10, cy - 10), textcoords='data',
#                         color='white', fontsize=9, weight='bold',
#                         arrowprops=dict(arrowstyle="->", color=l_color, lw=1.5),
#                         bbox=dict(boxstyle="square,pad=0.1", fc=l_color, alpha=0.6))

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     output_png = os.path.join(predictions_dir, "Smart_Montage.png")
#     plt.savefig(output_png)
#     print(f"🖼️ Clean Montage saved to: {output_png}")
#     plt.show()

# # --- MAIN EXECUTION ---
# if os.path.exists(file_t1) and os.path.exists(file_t2):
#     print(f"📂 Analyzing...")
#     fixed_t2_data, affine_ref = resample_to_match(file_t2, file_t1)
    
#     # Run Logic
#     labeled_b, labeled_f, report, _ = track_longitudinal(file_t1, nib.Nifti1Image(fixed_t2_data, affine_ref))
    
#     # 1. Visualization (Smart Montage)
#     visualize_smart_montage(labeled_f, report['New_IDs'])
    
#     # 2. Print Summary
#     print("\n" + "="*50)
#     print(f"   🩺 FINAL SUMMARY: Patient 01")
#     print("="*50)
#     print(f"   Baseline Lesions:     {report['Baseline_Count']}")
#     print(f"   Follow-up Lesions:    {report['Followup_Count']}")
#     print("-" * 50)
#     print(f"   🔴 NEW Lesions Detected:      {report['New_Lesions']}")
#     print(f"   🟢 Lesions Disappeared:       {report['Disappeared_Lesions']}")
#     print("="*50)
#     print(f"   Report saved to: {os.path.join(predictions_dir, 'Final_Report.csv')}")
    
#     # Save CSV
#     pd.DataFrame([report]).to_csv(os.path.join(predictions_dir, "Final_Report.csv"), index=False)
    
# else:
#     print("❌ Error: Files not found.")







































# import os
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom
# from lesion_tracker import track_longitudinal

# # --- CONFIGURATION ---
# predictions_dir = r"../Outputs"
# file_t1 = os.path.join(predictions_dir, "Case01_S1.nii")
# file_t2 = os.path.join(predictions_dir, "Case01_S2.nii")

# # Handle extensions
# if not os.path.exists(file_t1) and os.path.exists(file_t1 + ".gz"):
#     file_t1 += ".gz"
# if not os.path.exists(file_t2) and os.path.exists(file_t2 + ".gz"):
#     file_t2 += ".gz"

# def resample_to_match(target_img_path, reference_img_path):
#     ref_img = nib.load(reference_img_path)
#     tgt_img = nib.load(target_img_path)
    
#     ref_data = ref_img.get_fdata()
#     tgt_data = tgt_img.get_fdata()
    
#     if ref_data.shape == tgt_data.shape:
#         return tgt_img.get_fdata(), tgt_img.affine
    
#     print(f"⚠️ Resizing Follow-up from {tgt_data.shape} to match Baseline...")
#     factors = [r / t for r, t in zip(ref_data.shape, tgt_data.shape)]
#     resized_data = zoom(tgt_data, factors, order=0)
#     return resized_data, ref_img.affine

# def plot_combined_scene(data_b, data_f, new_ids, slice_idx, sizes_dict):
#     """ 
#     Displays the requested 3-Plot Comparison AND Labels with Sizes.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # Helper to draw labels + Size
#     def draw_labels(ax, slice_data, is_followup=False):
#         unique_ids = np.unique(slice_data)
#         for uid in unique_ids:
#             if uid == 0: continue
            
#             # Find center
#             y, x = np.where(slice_data == uid)
#             cy, cx = np.mean(y), np.mean(x)
            
#             # Determine Color and Text
#             is_new = (uid in new_ids) and is_followup
#             color = 'red' if is_new else 'lime' # Lime green is high contrast
            
#             # Get size (if available in follow-up)
#             size_txt = ""
#             if is_followup and uid in sizes_dict:
#                 size_txt = f"\n{sizes_dict[uid]}px"
            
#             # Draw Text
#             ax.text(cx, cy, f"{int(uid)}{size_txt}", color=color, fontsize=9, weight='bold',
#                     ha='center', va='center')

#     # 1. Baseline Plot
#     axes[0].imshow(data_b[:, :, slice_idx], cmap='gray', interpolation='none')
#     axes[0].set_title(f"Baseline (T1) - Slice {slice_idx}", fontsize=12)
#     axes[0].axis('off')
#     draw_labels(axes[0], data_b[:, :, slice_idx], is_followup=False)
    
#     # 2. Follow-up Plot
#     axes[1].imshow(data_f[:, :, slice_idx], cmap='gray', interpolation='none')
#     axes[1].set_title(f"Follow-up (T2) - Slice {slice_idx}", fontsize=12)
#     axes[1].axis('off')
#     draw_labels(axes[1], data_f[:, :, slice_idx], is_followup=True)
    
#     # 3. Clinical "Change" Map
#     h, w = data_f.shape[:2]
#     rgb_map = np.zeros((h, w, 3))
#     slice_f = data_f[:, :, slice_idx]
    
#     unique_ids = np.unique(slice_f)
#     for lesion_id in unique_ids:
#         if lesion_id == 0: continue
#         mask = (slice_f == lesion_id)
#         if lesion_id in new_ids:
#             rgb_map[mask] = [1, 0, 0] # RED
#         else:
#             rgb_map[mask] = [0, 0, 1] # BLUE
            
#     axes[2].imshow(rgb_map)
#     axes[2].set_title("Clinical Map\n(Red=New, Blue=Stable)", fontsize=12)
#     axes[2].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# # --- MAIN EXECUTION ---
# if os.path.exists(file_t1) and os.path.exists(file_t2):
#     print(f"📂 Analyzing...")
#     fixed_t2_data, affine_ref = resample_to_match(file_t2, file_t1)
    
#     # Run Logic
#     labeled_b, labeled_f, report, _ = track_longitudinal(file_t1, nib.Nifti1Image(fixed_t2_data, affine_ref))
    
#     # --- PRINT CENSUS (To find Number 1) ---
#     print("\n" + "="*40)
#     print("   🕵️‍♀️ LESION CENSUS (Top 5 Highest)")
#     print("="*40)
#     print("   ID  | Slice (Height) | Size (Voxels)")
#     print("   --- | -------------- | -------------")
    
#     # Scan the 3D volume to find where lesions live
#     for i in range(1, 6): # Print top 5 IDs
#         coords = np.where(labeled_f == i)
#         if len(coords[0]) > 0:
#             z_slice = int(np.mean(coords[2])) # Average Z position
#             size = report['Sizes_T2'].get(i, "N/A")
#             print(f"   #{i}  | Slice {z_slice:<10} | {size}")
#         else:
#             print(f"   #{i}  | (Disappeared or not in T2)")
            
#     # Visualization
#     # Auto-pick the slice with the MOST lesions to show the "Busy" view
#     # best_slice = np.argmax(np.sum(labeled_f > 0, axis=(0, 1)))
#     # best_slice = 10
#     best_slice = 25
#     print(f"\n🖼️ Displaying Slice {best_slice} (This slice has the most activity)")
    
#     plot_combined_scene(labeled_b, labeled_f, report['New_IDs'], best_slice, report['Sizes_T2'])
    
# else:
#     print("❌ Error: Files not found.")



















































# import os
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom
# from lesion_tracker import track_longitudinal

# # --- CONFIGURATION ---
# predictions_dir = r"../Outputs"
# file_t1 = os.path.join(predictions_dir, "Case01_S1.nii.gz")
# file_t2 = os.path.join(predictions_dir, "Case01_S2.nii.gz")

# # Handle extensions
# if not os.path.exists(file_t1) and os.path.exists(file_t1.replace(".gz", "")):
#     file_t1 = file_t1.replace(".gz", "")
# if not os.path.exists(file_t2) and os.path.exists(file_t2.replace(".gz", "")):
#     file_t2 = file_t2.replace(".gz", "")

# def resample_to_match(target_img_path, reference_img_path):
#     ref_img = nib.load(reference_img_path)
#     tgt_img = nib.load(target_img_path)
#     ref_data = ref_img.get_fdata()
#     tgt_data = tgt_img.get_fdata()
    
#     if ref_data.shape == tgt_data.shape:
#         return tgt_img.get_fdata(), tgt_img.affine
    
#     print(f"⚠️ Resizing Follow-up from {tgt_data.shape} to match Baseline {ref_data.shape}...")
#     factors = [r / t for r, t in zip(ref_data.shape, tgt_data.shape)]
#     resized_data = zoom(tgt_data, factors, order=0)
#     return resized_data, ref_img.affine

# def plot_1_clinical_map(data_f, new_ids, slice_idx):
#     """ First Plot: Just the Red/Blue Map """
#     plt.figure(figsize=(8, 8))
    
#     # Create RGB Map
#     h, w = data_f.shape[:2]
#     rgb_map = np.zeros((h, w, 3))
#     slice_f = data_f[:, :, slice_idx]
    
#     unique_ids = np.unique(slice_f)
#     has_content = False
    
#     for lesion_id in unique_ids:
#         if lesion_id == 0: continue
#         has_content = True
#         mask = (slice_f == lesion_id)
#         if lesion_id in new_ids:
#             rgb_map[mask] = [1, 0, 0] # RED = New
#         else:
#             rgb_map[mask] = [0, 0, 1] # BLUE = Stable
            
#     plt.imshow(rgb_map)
#     plt.title(f"Clinical Status Map (Slice {slice_idx})\nRed = New Lesion | Blue = Stable Lesion", fontsize=14)
#     plt.axis('off')
    
#     print("🖼️ Displaying Plot 1: Clinical Map (Close window to see next plot)")
#     plt.show()

# def plot_2_detailed_labels(data_b, data_f, new_ids, slice_idx):
#     """ Second Plot: Detailed IDs """
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
#     # Helper to draw sharp labels
#     def draw_labels(ax, slice_data, title):
#         ax.imshow(slice_data, cmap='gray')
#         ax.set_title(title, fontsize=14, weight='bold')
#         ax.axis('off')
        
#         unique_ids = np.unique(slice_data)
#         for uid in unique_ids:
#             if uid == 0: continue
            
#             # Find center
#             y, x = np.where(slice_data == uid)
#             cy, cx = np.mean(y), np.mean(x)
            
#             # Draw Box and Text for high visibility
#             label_text = str(int(uid))
#             color = 'red' if uid in new_ids else '#00FF00' # Bright Green for stable
            
#             ax.text(cx, cy, label_text, color='white', fontsize=12, weight='bold',
#                     ha='center', va='center',
#                     bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="white", alpha=0.8))

#     # Baseline
#     draw_labels(axes[0], data_b[:, :, slice_idx], "Baseline (T1)\nLabels Ordered Top-to-Bottom")
    
#     # Follow-up
#     draw_labels(axes[1], data_f[:, :, slice_idx], "Follow-up (T2)\n(IDs Persist)")
    
#     print("🖼️ Displaying Plot 2: Detailed Labels")
#     plt.show()

# # --- MAIN EXECUTION ---
# if os.path.exists(file_t1) and os.path.exists(file_t2):
#     print(f"📂 Analyzing...")
    
#     # 1. Fix Size
#     fixed_t2_data, affine_ref = resample_to_match(file_t2, file_t1)
#     fixed_t2_path = file_t2.replace(".nii", "_fixed.nii")
#     new_img = nib.Nifti1Image(fixed_t2_data, affine_ref)
#     nib.save(new_img, fixed_t2_path)
    
#     # 2. Run Logic
#     labeled_b, labeled_f, report, _ = track_longitudinal(file_t1, fixed_t2_path)
    
#     # 3. Print Stats
#     print("\n" + "="*40)
#     print(f"   Final Count: {report['Followup_Count']} Lesions")
#     print(f"   New: {report['New_Lesions']} | Gone: {report['Disappeared_Lesions']}")
#     print("="*40)

#     # 4. Visualization (Two Stages)
#     # Find best slice (most activity)
#     best_slice = np.argmax(np.sum(labeled_f > 0, axis=(0, 1)))
    
#     # Plot 1: Clinical Map
#     plot_1_clinical_map(labeled_f, report['New_IDs'], best_slice)
    
#     # Plot 2: Detailed Labels (Shows after you close Plot 1)
#     plot_2_detailed_labels(labeled_b, labeled_f, report['New_IDs'], best_slice)
    
# else:
#     print("❌ Error: Files not found.")













































# import os
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import matplotlib.pyplot as plt
# from scipy.ndimage import zoom
# from lesion_tracker import track_longitudinal

# # --- CONFIGURATION ---
# predictions_dir = r"../Outputs"
# # We define the paths to the files you downloaded
# file_t1 = os.path.join(predictions_dir, "Case01_S1.nii")
# file_t2 = os.path.join(predictions_dir, "Case01_S2.nii")

# # Handle hidden extensions if necessary
# if not os.path.exists(file_t1) and os.path.exists(file_t1 + ".gz"):
#     file_t1 += ".gz"
# if not os.path.exists(file_t2) and os.path.exists(file_t2 + ".gz"):
#     file_t2 += ".gz"

# def resample_to_match(target_img_path, reference_img_path):
#     """
#     Forces target_img to have exactly the same shape as reference_img.
#     Uses Nearest Neighbor interpolation to keep labels (0, 1, 2) intact.
#     """
#     ref_img = nib.load(reference_img_path)
#     tgt_img = nib.load(target_img_path)
    
#     ref_data = ref_img.get_fdata()
#     tgt_data = tgt_img.get_fdata()
    
#     if ref_data.shape == tgt_data.shape:
#         print("   Sizes match perfectly. No resizing needed.")
#         return tgt_img.get_fdata(), tgt_img.affine # Return data and affine directly
    
#     print(f"⚠️ Resizing Follow-up from {tgt_data.shape} to match Baseline {ref_data.shape}...")
    
#     # Calculate zoom factors for each dimension (x, y, z)
#     factors = [r / t for r, t in zip(ref_data.shape, tgt_data.shape)]
    
#     # Resize (order=0 means 'nearest neighbor' to preserve integer labels)
#     resized_data = zoom(tgt_data, factors, order=0)
    
#     return resized_data, ref_img.affine


# def visualize_results(data_b, data_f, new_ids):
#     """
#     Creates a visual report with NUMBER LABELS drawn on the lesions.
#     """
#     # Find slice with most lesions in Follow-up
#     slice_idx = np.argmax(np.sum(data_f > 0, axis=(0, 1)))

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
#     # --- HELPER TO DRAW LABELS ---
#     def add_labels(ax, slice_data):
#         """Finds center of each lesion in 2D and writes its ID."""
#         unique_ids = np.unique(slice_data)
#         for uid in unique_ids:
#             if uid == 0: continue # Skip background
            
#             # Find coordinates of this lesion
#             y_coords, x_coords = np.where(slice_data == uid)
#             cy, cx = np.mean(y_coords), np.mean(x_coords)
            
#             # specific logic: New lesions are Red, others Yellow text
#             color = 'red' if uid in new_ids else 'yellow'
            
#             # Write the number
#             ax.text(cx, cy, str(int(uid)), color=color, fontsize=10, 
#                     ha='center', va='center', weight='bold')

#     # 1. Baseline Plot
#     axes[0].imshow(data_b[:, :, slice_idx], cmap='gray', interpolation='none')
#     axes[0].set_title(f"Baseline (T1) - Slice {slice_idx}\n(Yellow IDs = Stable)")
#     axes[0].axis('off')
#     add_labels(axes[0], data_b[:, :, slice_idx])
    
#     # 2. Follow-up Plot (Raw)
#     axes[1].imshow(data_f[:, :, slice_idx], cmap='gray', interpolation='none')
#     axes[1].set_title(f"Follow-up (T2) - Slice {slice_idx}\n(ID Persistence Check)")
#     axes[1].axis('off')
#     add_labels(axes[1], data_f[:, :, slice_idx])
    
#     # 3. Clinical "Change" Map
#     h, w = data_f.shape[:2]
#     rgb_map = np.zeros((h, w, 3))
    
#     slice_f = data_f[:, :, slice_idx]
#     unique_ids = np.unique(slice_f)
    
#     for lesion_id in unique_ids:
#         if lesion_id == 0: continue
        
#         mask = (slice_f == lesion_id)
#         if lesion_id in new_ids:
#             rgb_map[mask] = [1, 0, 0] # RED for New
#         else:
#             rgb_map[mask] = [0, 0, 1] # BLUE for Stable
            
#     axes[2].imshow(rgb_map)
#     axes[2].set_title("Clinical Map\n(Red=New, Blue=Stable)")
#     axes[2].axis('off')
    
#     # Add numbers to the colored map too
#     add_labels(axes[2], slice_f)
    
#     plt.tight_layout()
#     output_png = os.path.join(predictions_dir, "Clinical_Visualization_Labeled.png")
#     plt.savefig(output_png)
#     print(f"🖼️ Labeled Visualization saved to: {output_png}")
#     plt.show()








# # def visualize_results(data_b, data_f, new_ids):
# #     """
# #     Creates a visual report: Baseline vs Followup.
# #     Highlights NEW lesions in RED.
# #     """
# #     # Find the slice with the most lesions to show something interesting
# #     slice_idx = np.argmax(np.sum(data_f, axis=(0, 1)))

# #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
# #     # 1. Baseline Plot
# #     axes[0].imshow(data_b[:, :, slice_idx], cmap='gray', interpolation='none')
# #     axes[0].set_title(f"Baseline (T1) - Slice {slice_idx}")
# #     axes[0].axis('off')
    
# #     # 2. Follow-up Plot (Raw)
# #     axes[1].imshow(data_f[:, :, slice_idx], cmap='gray', interpolation='none')
# #     axes[1].set_title(f"Follow-up (T2) - Slice {slice_idx}")
# #     axes[1].axis('off')
    
# #     # 3. Clinical "Change" Map
# #     # Create an RGB image: Blue = Stable, Red = New
# #     h, w = data_f.shape[:2]
# #     rgb_map = np.zeros((h, w, 3))
    
# #     # Get mask of this slice
# #     slice_f = data_f[:, :, slice_idx]
    
# #     # Color logic
# #     unique_ids = np.unique(slice_f)
# #     for lesion_id in unique_ids:
# #         if lesion_id == 0: continue
        
# #         mask = (slice_f == lesion_id)
# #         if lesion_id in new_ids:
# #             rgb_map[mask] = [1, 0, 0] # RED for New
# #         else:
# #             rgb_map[mask] = [0, 0, 1] # BLUE for Stable
            
# #     axes[2].imshow(rgb_map)
# #     axes[2].set_title("Clinical Map (Red=New, Blue=Stable)")
# #     axes[2].axis('off')
    
# #     plt.tight_layout()
# #     output_png = os.path.join(predictions_dir, "Clinical_Visualization.png")
# #     plt.savefig(output_png)
# #     print(f"🖼️ Visualization saved to: {output_png}")
# #     plt.show()

# # --- MAIN EXECUTION ---
# if os.path.exists(file_t1) and os.path.exists(file_t2):
#     print(f"📂 Analyzing Patient 1...")
    
#     # 1. Fix the Size Mismatch (Crucial Step)
#     # This function returns the resized DATA (numpy array), not the file object
#     fixed_t2_data, affine_ref = resample_to_match(file_t2, file_t1)
    
#     # We need to save this fixed data to a temp file so the tracker can load it
#     fixed_t2_path = file_t2.replace(".nii", "_fixed.nii")
#     new_img = nib.Nifti1Image(fixed_t2_data, affine_ref)
#     nib.save(new_img, fixed_t2_path)
    
#     # 2. Run Tracking on the FIXED file path
#     labeled_b, labeled_f, report, _ = track_longitudinal(file_t1, fixed_t2_path)
    
#     # 3. Print Report
#     print("\n" + "="*50)
#     print(f"   🩺 FINAL REPORT: Patient 01")
#     print("="*50)
#     print(f"   Baseline Count:       {report['Baseline_Count']}")
#     print(f"   Follow-up Count:      {report['Followup_Count']}")
#     print("-" * 50)
#     print(f"   🔴 NEW Lesions:       {report['New_Lesions']}")
#     print(f"   🟢 Gone Lesions:      {report['Disappeared_Lesions']}")
#     print("="*50)

#     # 4. Generate Visualization
#     visualize_results(labeled_b, labeled_f, report['New_IDs'])
    
#     # 5. Save Report
#     output_csv = os.path.join(predictions_dir, "Final_Report.csv")
#     pd.DataFrame([report]).to_csv(output_csv, index=False)
#     print(f"\n✅ Official Report saved to: {output_csv}")
    
# else:
#     print("❌ Error: Files not found. Check filenames in Outputs folder.")