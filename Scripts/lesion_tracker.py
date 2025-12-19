import numpy as np
import nibabel as nib
from scipy.ndimage import label, center_of_mass
from skimage.morphology import remove_small_objects






def get_sorted_labels(binary_mask, min_voxels=20):
    """
    Sorts 3D lesions Top-to-Bottom (High Z -> Low Z).
    Returns the labeled mask and a dictionary of {ID: Size_in_Voxels}.
    """
    # 1. Clean Noise 
    """ our "remove_small_objects" scans the 3D volume. If it finds a cluster of "on" pixels smaller than min_voxels (20),
      it flips them to 0 (background)"""
    bool_mask = binary_mask.astype(bool)
    clean_mask = remove_small_objects(bool_mask, min_size=min_voxels, connectivity=2)
    
    # 2. Label Components
    """ connected components Analysis: Each distinct lesion gets a unique integer ID """
    labeled_map, num_features = label(clean_mask, structure=np.ones((3,3,3)))
    
    if num_features == 0:
        return labeled_map, 0, [], {}

    # 3. Get Center of mass of every lesion (Z, Y, X) to sort based on location
    coms = center_of_mass(clean_mask, labeled_map, range(1, num_features + 1))
    """Lesion 1 is at (35, 100, 50), Lesion 2 is at (10, 200, 200)"""
    
    # that's a list where we have the data about each lesion >> ((Old_ID, Z, Y, X, Size))
    lesion_info = []
    for idx, com in enumerate(coms):
        old_id = idx + 1
        z, y, x = com[2], com[1], com[0]
        
        # Calculate Size in voxels (Volume)
        size = np.sum(labeled_map == old_id)
        
        # Store (Old_ID, Z, Y, X, Size)
        lesion_info.append((old_id, z, y, x, size))
    


    # 4.Sort Key: First by Z (Height), then Y (Front/Back), then X (Left/Right)
        # reverse=True means descending order
        # key=k[1]: This tells Python to look at the Z-coordinate (height) first.
    lesion_info.sort(key=lambda k: (k[1], k[2], k[3]), reverse=True)
    
    # 5. Re-Label
    new_mask = np.zeros_like(labeled_map)
    active_ids = []
    size_dict = {} # dictonary (key and value) to hold sizes of each lesion
    
    # mapping the old IDs to new sorted IDs to build the new labeld mask
    for new_id, (old_id, z, y, x, size) in enumerate(lesion_info, start=1 ): #start counting from 1 not 0
        new_mask[labeled_map == old_id] = new_id
        active_ids.append(new_id)
        # mapping the ID to its size
        size_dict[new_id] = size
        
        # 3D map, integer, list, dictionary
    return new_mask, num_features, active_ids, size_dict
# where num_features is the Total count of lesions found. (number)








def track_longitudinal(baseline_path, followup_path, min_size=20, silence=False):
    # --- 1. Load & Process Images ---
    img_b = nib.load(baseline_path)
    data_b = img_b.get_fdata().astype(int)
    
    if isinstance(followup_path, str):
        img_f = nib.load(followup_path)
        data_f = img_f.get_fdata().astype(int)
    else:
        img_f = followup_path
        data_f = img_f.get_fdata().astype(int)

    if not silence: print(f"   Analysing Baseline...")
    labeled_b, max_id_b, ids_b, sizes_b = get_sorted_labels(data_b, min_voxels=min_size)
    
    if not silence: print(f"   Analysing Follow-up...")
    labeled_f_temp, num_f, ids_f, sizes_f_temp = get_sorted_labels(data_f, min_voxels=min_size)

    # --- 2. Build the "Association Map" (Who touches Who?) ---
    # Dictionary structure: { T2_ID: [List_of_overlapping_T1_IDs] }
    t2_to_t1_map = {}
    
    # Reverse map for splitting: { T1_ID: [List_of_overlapping_T2_IDs] }
    t1_to_t2_map = {id_b: [] for id_b in ids_b} 

    for t2_id in ids_f:
        # Isolate T2 lesion
        lesion_mask = (labeled_f_temp == t2_id)
        
        # Check what exists at these coordinates in T1
        overlap_values = labeled_b[lesion_mask]
        
        # Find unique T1 IDs involved (exclude 0/background)
        unique_t1_ids = np.unique(overlap_values)
        unique_t1_ids = unique_t1_ids[unique_t1_ids != 0]
        
        t2_to_t1_map[t2_id] = list(unique_t1_ids)
        
        # Populate reverse map
        for t1_id in unique_t1_ids:
            t1_to_t2_map[t1_id].append(t2_id)

    # --- 3. Classification Logic ---
    final_labeled_f = np.zeros_like(labeled_f_temp)
    report_details = [] # To store detailed status for CSV
    
    # Counters
    stats = {
        "New": 0, "Stable": 0, "Enlarged": 0, "Shrunk": 0, 
        "Merged": 0, "Split": 0, "Disappeared": 0
    }
    
    # Track which IDs we have used to calculate Disappeared later
    matched_t1_ids = set()

    for t2_id in ids_f:
        current_size = sizes_f_temp[t2_id]
        overlapping_t1s = t2_to_t1_map[t2_id]
        
        status = "Unknown"
        assigned_id = 0
        
        # CASE A: NEW (No overlap with any T1 lesion)
        if len(overlapping_t1s) == 0:
            status = "New"
            max_id_b += 1
            assigned_id = max_id_b
            stats["New"] += 1
            
        # CASE B: MERGE (One T2 touches Multiple T1s)
        elif len(overlapping_t1s) > 1:
            status = f"Merged (from {overlapping_t1s})"
            # Inherit ID of the LARGEST Baseline lesion involved
            # This keeps color consistent with the main parent
            sizes_of_parents = [sizes_b[pid] for pid in overlapping_t1s]
            main_parent = overlapping_t1s[np.argmax(sizes_of_parents)]
            assigned_id = main_parent
            stats["Merged"] += 1
            matched_t1_ids.update(overlapping_t1s)

        # CASE C: ONE-TO-ONE or SPLIT
        else:
            # We touch exactly one T1 lesion. 
            # But does that T1 lesion touch ONLY us? Or others too?
            t1_parent = overlapping_t1s[0]
            siblings = t1_to_t2_map[t1_parent]
            
            # CASE C1: SPLIT (The parent T1 touches multiple T2s)
            if len(siblings) > 1:
                status = f"Split (from {t1_parent})"
                assigned_id = t1_parent # They all inherit the parent ID (visualizing fragmentation)
                
                # Only count "Split" once per group to avoid double counting stats? 
                # Or count every fragment? Let's count every fragment as a "Split Result"
                stats["Split"] += 1
                matched_t1_ids.add(t1_parent)

            # CASE C2: PURE STABLE (1-to-1 Match)
            else:
                # Check Size Change
                old_size = sizes_b[t1_parent]
                
                # 25% Increase Logic
                if current_size > (old_size * 1.25):
                    status = "Enlarged"
                    stats["Enlarged"] += 1
                # Shrink Logic (Any decrease? Or define a threshold?)
                elif current_size < old_size:
                    status = "Shrunk"
                    stats["Shrunk"] += 1
                else:
                    status = "Stable"
                    stats["Stable"] += 1
                
                assigned_id = t1_parent
                matched_t1_ids.add(t1_parent)

        # Apply ID and Store Data
        final_labeled_f[labeled_f_temp == t2_id] = assigned_id
        
        report_details.append({
            "T2_ID": assigned_id, # The final ID on the map
            "Original_T2_ID": t2_id, # The raw ID before sorting
            "Status": status,
            "Size_T2": current_size,
            "Size_T1_Ref": sizes_b[overlapping_t1s[0]] if len(overlapping_t1s)==1 else "N/A"
        })

    # --- 4. Disappearance Logic ---
    disappeared_ids = list(set(ids_b) - matched_t1_ids)
    stats["Disappeared"] = len(disappeared_ids)

# [NEW] Add Disappeared Lesions to the Detailed Report
    for gone_id in disappeared_ids:
        report_details.append({
            "T2_ID": "N/A",           # Doesn't exist in T2
            "Tracking_ID": gone_id,   # The ID we were tracking (from Baseline)
            "Status": "Disappeared",
            "Size_T2": 0,
            "Size_T1_Ref": sizes_b[gone_id]
        })
        
    # [NEW] Update the "Tracking_ID" for existing lesions too (for consistency)
    # This helps sorting the final report
    for item in report_details:
        if "Tracking_ID" not in item:
            # If it's New, use its T2_ID. If it's Stable/Split, use the T2_ID (which inherited the T1 ID)
            item["Tracking_ID"] = item["T2_ID"]

    list_of_new_ids = [item['T2_ID'] for item in report_details if item['Status'] == 'New']

    # --- 5. Generate Report Dictionary ---
    final_report = {
            "Baseline_Count": len(ids_b),
            "Followup_Count": num_f,
            "New": stats["New"],
            "Disappeared": stats["Disappeared"],
            "Stable_Exact": stats["Stable"],
            "Enlarged_GT25": stats["Enlarged"],
            "Shrunk": stats["Shrunk"],
            "Merged_Events": stats["Merged"],
            "Split_Fragments": stats["Split"],
            
            # [FIX] Add this key back so the visualizer works
            "New_IDs": list_of_new_ids, 
            
            "Details": report_details
        }

    return labeled_b, final_labeled_f, final_report, img_b.affine

















# def track_longitudinal(baseline_path, followup_path, min_size=20, silence=False):
#     # Load Baseline
#     img_b = nib.load(baseline_path)
#     data_b = img_b.get_fdata().astype(int)
    
#     # Load Follow-up (support both path and fixed image object)
#     if isinstance(followup_path, str):
#         img_f = nib.load(followup_path)
#         data_f = img_f.get_fdata().astype(int)
#     else:
#         # This happens if we ran the 'Resampling' fix earlier cuz it returns a Nifti object
#         img_f = followup_path
#         data_f = img_f.get_fdata().astype(int)


# # pass the images separately to get_sorted_labels function to get the labeled maps for each one
#     if not silence:
#         print(f"   Analysing Baseline...")
#     labeled_b, max_id_b, ids_b, sizes_b = get_sorted_labels(data_b, min_voxels=min_size)
    
#     if not silence:
#         print(f"   Analysing Follow-up...")
#     labeled_f_temp, num_f, _, sizes_f_temp = get_sorted_labels(data_f, min_voxels=min_size)
    


#     # --- Matching Logic ---
#     final_labeled_f = np.zeros_like(labeled_f_temp)
#     matched_ids = []
#     new_lesions = []
#     final_sizes = {} # Store sizes for the final report
    
#     for temp_id in range(1, num_f + 1):
#         # 1. Isolate one specific lesion from the T2 scan
#         lesion_mask = (labeled_f_temp == temp_id)
#         current_size = sizes_f_temp[temp_id]
        
#         # Check Overlap (here we look at the EXACT SAME coordinates in the T1 scan)
#         baseline_overlap = labeled_b[lesion_mask]
#         baseline_overlap = baseline_overlap[baseline_overlap > 0] # Ignore background (0) and care about the lesion
        
#         if len(baseline_overlap) > 0:
#             # Match
#             # Find which ID appears most often (Dominant overlap) and inherit it
#             """the argmax here handles edge cases where a new lesion might touch two old lesions.
#               It picks the one with the most overlap."""
#             inherited_id = np.bincount(baseline_overlap).argmax()
#              # Assign the OLD ID to the NEW map
#             final_labeled_f[lesion_mask] = inherited_id
#             matched_ids.append(inherited_id)
#             final_sizes[inherited_id] = current_size
#         else:
#             # New
#             # # Increment the highest known ID (e.g., 17 -> 18)
#             max_id_b += 1
#             new_id = max_id_b
#             final_labeled_f[lesion_mask] = new_id
#             new_lesions.append(new_id)
#             final_sizes[new_id] = current_size
            
#             """we detect the disappeared lesion by checking what was in the baseline but not in the matched list"""
#     disappeared_ids = list(set(ids_b) - set(matched_ids))
    
#     # here we prepare the final report dictionary 
#     report = {
#         "Baseline_Count": len(ids_b),
#         "Followup_Count": len(matched_ids) + len(new_lesions),
#         "New_Lesions": len(new_lesions),
#         "Disappeared_Lesions": len(disappeared_ids),
#         "New_IDs": new_lesions,
#         "Disappeared_IDs": disappeared_ids,
#         "Sizes_T2": final_sizes # Passing sizes to report
#     }
    
#     return labeled_b, final_labeled_f, report, img_b.affine






























# import numpy as np
# import nibabel as nib
# from scipy.ndimage import label, center_of_mass
# from skimage.morphology import remove_small_objects

# def get_sorted_labels(binary_mask, min_voxels=20):
#     """
#     Sorts lesions strictly:
#     1. Highest Z (Top of Brain) -> Gets ID 1
#     2. If Z is same, Highest Y (Top of Image) -> Gets ID 1
#     """
#     # 1. Clean Noise
#     bool_mask = binary_mask.astype(bool)
#     clean_mask = remove_small_objects(bool_mask, min_size=min_voxels, connectivity=2)
    
#     # 2. Label
#     labeled_map, num_features = label(clean_mask, structure=np.ones((3,3,3)))
    
#     if num_features == 0:
#         return labeled_map, 0, []

#     # 3. Calculate Centers (Z, Y, X)
#     coms = center_of_mass(clean_mask, labeled_map, range(1, num_features + 1))
    
#     lesion_info = []
#     for idx, com in enumerate(coms):
#         old_id = idx + 1
#         z = com[2] 
#         y = com[1]
#         x = com[0]
#         # Store (Old_ID, Z, Y, X)
#         lesion_info.append((old_id, z, y, x))
    
#     # 4. SORTING LOGIC (The Fix)
#     # We sort by Z (descending), then Y (descending), then X (descending)
#     # This ensures "Top of Brain" and "Top of Slice" get the first IDs.
#     lesion_info.sort(key=lambda k: (k[1], k[2], k[3]), reverse=True)
    
#     # 5. Re-Assign IDs
#     new_mask = np.zeros_like(labeled_map)
#     active_ids = []
    
#     for new_id, (old_id, z, y, x) in enumerate(lesion_info, start=1):
#         new_mask[labeled_map == old_id] = new_id
#         active_ids.append(new_id)
        
#     return new_mask, num_features, active_ids

# def track_longitudinal(baseline_path, followup_path, min_size=20):
#     # Load Images
#     img_b = nib.load(baseline_path)
#     data_b = img_b.get_fdata().astype(int)
    
#     # Handle the 'fixed' image object vs string path
#     if isinstance(followup_path, str):
#         img_f = nib.load(followup_path)
#         data_f = img_f.get_fdata().astype(int)
#     else:
#         img_f = followup_path
#         data_f = img_f.get_fdata().astype(int)

#     # Run Logic
#     print(f"   Analysing Baseline...")
#     labeled_b, max_id_b, ids_b = get_sorted_labels(data_b, min_voxels=min_size)
    
#     print(f"   Analysing Follow-up...")
#     labeled_f_temp, num_f, _ = get_sorted_labels(data_f, min_voxels=min_size)
    
#     # Match Logic
#     final_labeled_f = np.zeros_like(labeled_f_temp)
#     matched_ids = []
#     new_lesions = []
    
#     for temp_id in range(1, num_f + 1):
#         lesion_mask = (labeled_f_temp == temp_id)
#         baseline_overlap = labeled_b[lesion_mask]
#         baseline_overlap = baseline_overlap[baseline_overlap > 0]
        
#         if len(baseline_overlap) > 0:
#             # Overlap = Match
#             inherited_id = np.bincount(baseline_overlap).argmax()
#             final_labeled_f[lesion_mask] = inherited_id
#             matched_ids.append(inherited_id)
#         else:
#             # No Overlap = New
#             max_id_b += 1
#             new_id = max_id_b
#             final_labeled_f[lesion_mask] = new_id
#             new_lesions.append(new_id)
            
#     disappeared_ids = list(set(ids_b) - set(matched_ids))
    
#     report = {
#         "Baseline_Count": len(ids_b),
#         "Followup_Count": len(matched_ids) + len(new_lesions),
#         "New_Lesions": len(new_lesions),
#         "Disappeared_Lesions": len(disappeared_ids),
#         "New_IDs": new_lesions,
#         "Disappeared_IDs": disappeared_ids
#     }
#     return labeled_b, final_labeled_f, report, img_b.affine

















# import numpy as np
# import pandas as pd
# import nibabel as nib
# from scipy.ndimage import label, center_of_mass
# from skimage.morphology import remove_small_objects # New library we need

# def get_sorted_labels(binary_mask, min_voxels=20):
#     """
#     1. Removes tiny noise specks (smaller than min_voxels).
#     2. Labels the remaining lesions.
#     3. Sorts them from Top to Bottom (High Z to Low Z).
#     """
#     # --- STEP 1: CLEANING NOISE ---
#     # We convert to boolean because remove_small_objects expects bool
#     bool_mask = binary_mask.astype(bool)
    
#     # Remove objects smaller than 'min_voxels' (e.g., 20 pixels)
#     # connectivity=2 ensures we consider diagonal pixels connected in 3D
#     clean_mask = remove_small_objects(bool_mask, min_size=min_voxels, connectivity=2)
    
#     # --- STEP 2: LABELING ---
#     # structure=np.ones((3,3,3)) defines 3D connectivity
#     labeled_map, num_features = label(clean_mask, structure=np.ones((3,3,3)))
    
#     if num_features == 0:
#         return labeled_map, 0, []

#     # --- STEP 3: SORTING (UP to DOWN) ---
#     # Calculate center of mass for every lesion
#     coms = center_of_mass(clean_mask, labeled_map, range(1, num_features + 1))
    
#     lesion_info = []
#     for idx, com in enumerate(coms):
#         old_id = idx + 1
#         z_height = com[2] # We assume Z is the last dimension
#         lesion_info.append((old_id, z_height))
    
#     # Sort by Z Height (Descending = Top to Bottom)
#     lesion_info.sort(key=lambda x: x[1], reverse=True)
    
#     # --- STEP 4: RE-ASSIGN IDS ---
#     new_mask = np.zeros_like(labeled_map)
#     active_ids = []
    
#     for new_id, (old_id, z) in enumerate(lesion_info, start=1):
#         new_mask[labeled_map == old_id] = new_id
#         active_ids.append(new_id)
        
#     return new_mask, num_features, active_ids

# def track_longitudinal(baseline_path, followup_path, min_size=20):
#     """
#     Compares two scans with noise filtering enabled.
#     """
#     # Load Images
#     img_b = nib.load(baseline_path)
#     data_b = img_b.get_fdata().astype(int)
    
#     # Try loading as simple array first to handle fixed files
#     if isinstance(followup_path, str):
#         img_f = nib.load(followup_path)
#         data_f = img_f.get_fdata().astype(int)
#     else:
#         # Assuming it's already a Nifti image object (from our fix script)
#         img_f = followup_path
#         data_f = img_f.get_fdata().astype(int)

#     print(f"   Analysing Baseline (Filtering noise < {min_size} voxels)...")
#     labeled_b, max_id_b, ids_b = get_sorted_labels(data_b, min_voxels=min_size)
    
#     print(f"   Analysing Follow-up (Filtering noise < {min_size} voxels)...")
#     labeled_f_temp, num_f, _ = get_sorted_labels(data_f, min_voxels=min_size)
    
#     # --- MATCHING LOGIC ---
#     final_labeled_f = np.zeros_like(labeled_f_temp)
#     matched_ids = []
#     new_lesions = []
    
#     for temp_id in range(1, num_f + 1):
#         lesion_mask = (labeled_f_temp == temp_id)
        
#         # Check Overlap in Baseline
#         baseline_overlap = labeled_b[lesion_mask]
#         baseline_overlap = baseline_overlap[baseline_overlap > 0]
        
#         if len(baseline_overlap) > 0:
#             # Match Found!
#             inherited_id = np.bincount(baseline_overlap).argmax()
#             final_labeled_f[lesion_mask] = inherited_id
#             matched_ids.append(inherited_id)
#         else:
#             # New Lesion Found!
#             max_id_b += 1
#             new_id = max_id_b
#             final_labeled_f[lesion_mask] = new_id
#             new_lesions.append(new_id)
            
#     disappeared_ids = list(set(ids_b) - set(matched_ids))
    
#     report = {
#         "Baseline_Count": len(ids_b),
#         "Followup_Count": len(matched_ids) + len(new_lesions),
#         "New_Lesions": len(new_lesions),
#         "Disappeared_Lesions": len(disappeared_ids),
#         "New_IDs": new_lesions,
#         "Disappeared_IDs": disappeared_ids
#     }
    
#     return labeled_b, final_labeled_f, report, img_b.affine




















# import numpy as np
# import pandas as pd
# import nibabel as nib
# from scipy.ndimage import label, center_of_mass

# def get_sorted_labels(binary_mask):
#     """
#     Takes a raw binary mask (0s and 1s).
#     Finds lesions.
#     Sorts them from Top (High Z) to Bottom (Low Z).
#     Returns a labeled mask where 1=Topmost lesion, 2=Next down, etc.
#     """
#     # 1. Label connected components (Find individual blobs)
#     # structure=np.ones((3,3,3)) ensures diagonal pixels count as connected
#     labeled_map, num_features = label(binary_mask, structure=np.ones((3,3,3)))
    
#     if num_features == 0:
#         return labeled_map, 0, []

#     # 2. Calculate center of mass for every lesion
#     # This gives us (z, y, x) coordinates for labels 1, 2, 3...
#     coms = center_of_mass(binary_mask, labeled_map, range(1, num_features + 1))
    
#     # 3. Create a list of (Old_ID, Z_Height)
#     # Note: coms[i][2] is usually Z in NIfTI, but sometimes it's [0]. 
#     # We will assume Z is the last dimension (standard axial). 
#     # If your sorting looks wrong later, we switch this index.
#     lesion_info = []
#     for idx, com in enumerate(coms):
#         old_id = idx + 1
#         z_height = com[2] # Adjust index if Z-axis is different
#         lesion_info.append((old_id, z_height))
    
#     # 4. Sort by Z Height (Descending = Top to Bottom)
#     lesion_info.sort(key=lambda x: x[1], reverse=True)
    
#     # 5. Relabel the mask with the new sorted IDs
#     new_mask = np.zeros_like(labeled_map)
#     active_ids = []
    
#     for new_id, (old_id, z) in enumerate(lesion_info, start=1):
#         new_mask[labeled_map == old_id] = new_id
#         active_ids.append(new_id)
        
#     return new_mask, num_features, active_ids

# def track_longitudinal(baseline_path, followup_path):
#     """
#     Compares two scans. Enforces that IDs carry over.
#     """
#     # Load Images
#     img_b = nib.load(baseline_path)
#     data_b = img_b.get_fdata().astype(int)
    
#     img_f = nib.load(followup_path)
#     data_f = img_f.get_fdata().astype(int)
    
#     print(f"   Analysing Baseline...")
#     # 1. Process Baseline (Sort Top-to-Bottom)
#     labeled_b, max_id_b, ids_b = get_sorted_labels(data_b)
    
#     print(f"   Analysing Follow-up...")
#     # 2. Label Follow-up (Temporarily)
#     labeled_f_temp, num_f, _ = get_sorted_labels(data_f)
    
#     # 3. The Matching Logic
#     # We create a final mask for follow-up that respects Baseline IDs
#     final_labeled_f = np.zeros_like(labeled_f_temp)
    
#     matched_ids = []
#     new_lesions = []
    
#     # We assume any overlapping pixel means "Same Lesion"
#     # Iterate through each lesion found in the Follow-Up
#     for temp_id in range(1, num_f + 1):
#         # Create a boolean mask for just this one lesion
#         lesion_mask = (labeled_f_temp == temp_id)
        
#         # Check what value sits in this spot on the Baseline mask
#         # We look for the most common ID in the overlapping area
#         baseline_overlap = labeled_b[lesion_mask]
        
#         # Filter out 0 (background)
#         baseline_overlap = baseline_overlap[baseline_overlap > 0]
        
#         if len(baseline_overlap) > 0:
#             # Overlap found! It inherits the Old ID.
#             # (We take the most frequent overlapping ID to handle edge cases)
#             inherited_id = np.bincount(baseline_overlap).argmax()
#             final_labeled_f[lesion_mask] = inherited_id
#             matched_ids.append(inherited_id)
#         else:
#             # No overlap! This is a NEW lesion.
#             # Assign a brand new ID (starting from max_b + 1)
#             max_id_b += 1
#             new_id = max_id_b
#             final_labeled_f[lesion_mask] = new_id
#             new_lesions.append(new_id)
            
#     # 4. Determine Disappeared Lesions
#     # If an ID was in Baseline (ids_b) but NOT in our matched list, it's gone.
#     disappeared_ids = list(set(ids_b) - set(matched_ids))
    
#     # 5. Generate Report Dictionary
#     report = {
#         "Baseline_Count": len(ids_b),
#         "Followup_Count": len(matched_ids) + len(new_lesions),
#         "New_Lesions": len(new_lesions),
#         "Disappeared_Lesions": len(disappeared_ids),
#         "New_IDs": new_lesions,
#         "Disappeared_IDs": disappeared_ids
#     }
    
#     return labeled_b, final_labeled_f, report, img_b.affine

# # --- TEST BLOCK (Runs only if you click Play) ---
# if __name__ == "__main__":
#     # We will test this on Patient 1's GROUND TRUTH masks
#     # (Since we don't have model predictions yet, we use the real answers to test logic)
    
#     # Adjust path if needed based on your earlier 'check_data.py' result
#     p1_base = "../Raw_Data/Longitudinal_Tracking/patient01"
    
#     # Note: Using the GOLD STANDARD masks for testing logic
#     mask1 = f"{p1_base}/study1_FLAIR.nii.gz" # Using FLAIR just to test file reading? 
#     # WAIT! We need binary masks to test logic. 
#     # Let's use the 'gt.nii.gz' (Ground Truth) if available, or just threshold the FLAIR for a dummy test.
#     # Looking at your structure, you have 'gold_standard' or 'gt' folders? 
#     # Let's try to find the mask file. 
#     # Based on open_ms_data readme: longitudinal/raw/patient01/gt.nii.gz
    
#     mask1 = f"{p1_base}/gt.nii.gz"
#     # Actually, patient01 usually has ONE gt file for tracking changes, 
#     # but let's assume for this test we track Study1 vs Study2 FLAIR (thresholded)
#     # just to see if code runs.
    
#     print("⚠️ NOTE: To test this properly, we need binary masks.")
#     print("If you have 'mask1.nii.gz' and 'mask2.nii.gz', put their paths here.") 