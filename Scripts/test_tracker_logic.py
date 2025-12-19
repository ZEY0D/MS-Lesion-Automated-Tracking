import numpy as np
from lesion_tracker import get_sorted_labels

print("🧪 Testing Logic Module (Corrected Axis)...")

# 1. Create a Fake 3D Brain (Shape 10x10x10)
# NIfTI format is usually (X, Y, Z)
fake_mask = np.zeros((10, 10, 10))

# 2. Place Lesion A at the BOTTOM (Z=2)
# Notice we are now changing the THIRD number (index 2)
fake_mask[5, 5, 2] = 1 

# 3. Place Lesion B at the TOP (Z=8)
fake_mask[5, 5, 8] = 1

print("   Created fake brain with 2 lesions.")
print("   - Lesion A at Z=2 (Bottom)")
print("   - Lesion B at Z=8 (Top)")
print("   Doctor requires: Top lesion gets ID 1. Bottom gets ID 2.")

# 4. Run your sorting function
sorted_mask, count, ids = get_sorted_labels(fake_mask)

print(f"   Function found {count} lesions.")

# 5. Check ID at Top (Z=8)
id_at_top = sorted_mask[5, 5, 8]
print(f"   ID at Top (Z=8): {id_at_top} (Should be 1)")

# 6. Check ID at Bottom (Z=2)
id_at_bottom = sorted_mask[5, 5, 2]
print(f"   ID at Bottom (Z=2): {id_at_bottom} (Should be 2)")

if id_at_top == 1 and id_at_bottom == 2:
    print("✅ SUCCESS! Logic is logically distributed Up-to-Down.")
else:
    print(f"❌ FAILURE. Top got {id_at_top}, Bottom got {id_at_bottom}")