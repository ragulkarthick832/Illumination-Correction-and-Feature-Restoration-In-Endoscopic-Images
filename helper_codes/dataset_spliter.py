import os
import shutil
from sklearn.model_selection import train_test_split

input_folder = "../../Dataset/images"
output_base = "../../Dataset/gt"

# Create split folders
for split in ["train", "test", "validation"]:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

# Load image files
image_files = sorted([f for f in os.listdir(input_folder)
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))])

# Train 70%, Temp 30%
train_files, temp = train_test_split(image_files, test_size=0.30, random_state=42)

# Test 20%, Validation 10%
test_files, val_files = train_test_split(temp, test_size=0.3333, random_state=42)

splits = {
    "train": train_files,
    "test": test_files,
    "validation": val_files
}

# Copy images
for split_name, files in splits.items():
    for f in files:
        src = os.path.join(input_folder, f)
        dst = os.path.join(output_base, split_name, f)
        shutil.copy(src, dst)

print("✅ Image split completed successfully!")
