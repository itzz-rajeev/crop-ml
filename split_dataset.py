import os
import shutil
import random

# -------------------------
# Paths
# -------------------------
source_dir = "PlantVillage"  # Original dataset folder
train_dir = "dataset/train"
test_dir = "dataset/test"

# Create train/test folders if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratio
split_ratio = 0.8  # 80% train, 20% test

# Loop through each class folder
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        random.shuffle(images)
        
        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create class folders in train and test directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        # Copy images to respective folders
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, class_name, img))

print("✅ Dataset split into train and test folders successfully!")

