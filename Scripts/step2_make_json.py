import os
import json
import glob

# --- CONFIGURATION ---
# The folder where you just put the images
TARGET_FOLDER = r"../nnUNet_raw/Dataset501_MSLesion"
OUTPUT_FILE = os.path.join(TARGET_FOLDER, "dataset.json")

def generate_dataset_json():
    print(f"🕵️‍♀️ Scanning: {TARGET_FOLDER}")
    
    # 1. Find all training images
    imagesTr_dir = os.path.join(TARGET_FOLDER, "imagesTr")
    image_files = glob.glob(os.path.join(imagesTr_dir, "*.nii.gz"))
    
    if len(image_files) == 0:
        print("❌ Error: No images found! Did Step 2 work?")
        return

    # 2. Sort them so the list is consistent
    image_files.sort()
    
    # 3. Build the Training List
    training_list = []
    print(f"   Found {len(image_files)} images. Building index...")
    
    for img_path in image_files:
        # Get filename: "MS_P1_T1_0000.nii.gz"
        fname = os.path.basename(img_path)
        
        # Remove "_0000.nii.gz" to get the Case ID: "MS_P1_T1"
        case_id = fname.replace("_0000.nii.gz", "")
        
        # Add to list
        training_list.append({
            "image": f"./imagesTr/{case_id}.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })

    # 4. Define the Dataset Dictionary (The Rules)
    json_dict = {
        "name": "MS_Lesion_Tracking",
        "description": "Automated MS Lesion Segmentation for Dr. Makary",
        "reference": "MSLesSeg Dataset",
        "licence": "CC-BY-SA",
        "release": "1.0",
        "tensorImageSize": "3D",
        "modality": {
            "0": "FLAIR"   # This tells AI that Channel 0 is FLAIR
        },
        "labels": {
            "background": 0,
            "lesion": 1    # This tells AI that 1 is the target
        },
        "numTraining": len(training_list),
        "file_ending": ".nii.gz",
        "training": training_list
    }

    # 5. Save to File
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(json_dict, f, indent=4)
        
    print("-" * 30)
    print(f"✅ SUCCESS! Created dataset.json")
    print(f"   Location: {OUTPUT_FILE}")
    print(f"   Contains {len(training_list)} training cases.")

# Run it
generate_dataset_json()