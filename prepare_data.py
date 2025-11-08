import os
import shutil
import random

print("Starting to prepare data...")

# --- Configuration ---
BASE_DIR = 'dataset'
SPLIT_SIZE = 2500  # We will move 2,500 images for cats and 2,500 for dogs

# Define all the paths
train_dir = os.path.join(BASE_DIR, 'train')
validation_dir = os.path.join(BASE_DIR, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# --- Create validation directories if they don't exist ---
print("Making sure validation directories exist...")
os.makedirs(validation_cats_dir, exist_ok=True)
os.makedirs(validation_dogs_dir, exist_ok=True)

# --- Helper function to move files ---
def move_files(source_dir, dest_dir, file_list):
    """Moves a list of files from a source to a destination."""
    
    # --- FIX: This block is now indented correctly ---
    for file_name in file_list:
        source = os.path.join(source_dir, file_name)
        dest = os.path.join(dest_dir, file_name)
        # Check if file exists before moving
        if os.path.exists(source):
            shutil.move(source, dest)
        else:
            print(f"Warning: File {source} not found, skipping.")

# --- 1. Process CATS ---
print(f"Processing cat images...")
cat_files = os.listdir(train_cats_dir)
random.shuffle(cat_files)
cats_to_move = cat_files[:SPLIT_SIZE]
move_files(train_cats_dir, validation_cats_dir, cats_to_move)

# --- 2. Process DOGS ---
print(f"Processing dog images...")
dog_files = os.listdir(train_dogs_dir)
random.shuffle(dog_files)
dogs_to_move = dog_files[:SPLIT_SIZE]
move_files(train_dogs_dir, validation_dogs_dir, dogs_to_move)

print("Data preparation complete!")
print(f"Total cat images in train: {len(os.listdir(train_cats_dir))}")
print(f"Total cat images in validation: {len(os.listdir(validation_cats_dir))}")
print(f"Total dog images in train: {len(os.listdir(train_dogs_dir))}")
print(f"Total dog images in validation: {len(os.listdir(validation_dogs_dir))}")