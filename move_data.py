import os
import shutil

# Input directories
input_sq_dir = "augmented_squirrel_images"  # Directory with squirrel images
input_no_sq_dir = "lawn"                    # Directory with no squirrel images

# Output directories
dirs = ['data/train/squirrel', 'data/val/squirrel', 'data/train/no_squirrel', 'data/val/no_squirrel']
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

# Function to split and move images
def move_images(input_dir, output_train_dir, output_val_dir, label):
    images = [img for img in os.listdir(input_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
    total_len = len(images)
    val_num = total_len // 5  # 20% for validation
    train_num = total_len - val_num

    for idx, img in enumerate(images):
        img_path = os.path.join(input_dir, img)
        if idx < train_num:
            shutil.copy(img_path, os.path.join(output_train_dir, f"{label}_{idx}.jpg"))
        else:
            shutil.copy(img_path, os.path.join(output_val_dir, f"{label}_{idx}.jpg"))

# Move squirrel images
move_images(input_sq_dir, 'data/train/squirrel', 'data/val/squirrel', 'sq')

# Move no squirrel images
move_images(input_no_sq_dir, 'data/train/no_squirrel', 'data/val/no_squirrel', 'no_sq')
