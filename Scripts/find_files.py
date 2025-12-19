import os

# Start searching from the main project folder
search_dir = r"D:\MS_Lesion_Tracking"

print(f"🕵️ Searching for NIfTI files in: {search_dir} ...")
print("-" * 60)

found_count = 0

for root, dirs, files in os.walk(search_dir):
    for file in files:
        # Check for both .nii and .nii.gz
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            full_path = os.path.join(root, file)
            print(f"FOUND: {full_path}")
            found_count += 1

print("-" * 60)
if found_count == 0:
    print("❌ No .nii or .nii.gz files found! Check if the drive/folder is correct.")
else:
    print(f"✅ Found {found_count} files. Copy the correct paths from above!")