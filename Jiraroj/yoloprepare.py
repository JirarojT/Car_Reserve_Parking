import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
image_dir = 'generated_dataset'
train_dir = 'dataset/images/train'
val_dir = 'dataset/images/val'
label_dir = 'dataset/labels/train'
val_label_dir = 'dataset/labels/val'

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Get all image files and corresponding label files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
label_files = [f.replace('.jpg', '.txt') for f in image_files]

# Split data into train and validation sets (80% train, 20% validation)
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Move train images and labels
for img in train_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(train_dir, img))
    shutil.move(os.path.join(image_dir, img.replace('.jpg', '.txt')), os.path.join(label_dir, img.replace('.jpg', '.txt')))

# Move validation images and labels
for img in val_images:
    shutil.move(os.path.join(image_dir, img), os.path.join(val_dir, img))
    shutil.move(os.path.join(image_dir, img.replace('.jpg', '.txt')), os.path.join(val_label_dir, img.replace('.jpg', '.txt')))
