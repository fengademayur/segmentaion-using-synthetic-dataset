import os
import random
import shutil

# configuration
source_img_dir = r"/home/mayurf/main_tasks/kiros/kiros/synthetic_data/heavy_guy/imgs_with_bg"
source_mask_dir = r"/home/mayurf/main_tasks/kiros/kiros/synthetic_data/heavy_guy/masks_with_bg"


output_base_dir = r"/home/mayurf/main_tasks/kiros/kiros/synthetic_data/heavy_guy/dataset_split"

# Create  4 folders

train_img_out = os.path.join(output_base_dir, "train", "images")
train_mask_out = os.path.join(output_base_dir, "train", "masks")
val_img_out = os.path.join(output_base_dir, "val", "images")
val_mask_out = os.path.join(output_base_dir, "val", "masks")

for folder in [train_img_out, train_mask_out, val_img_out, val_mask_out]:
    os.makedirs(folder, exist_ok=True)

# shuffle data
all_images = [f for f in os.listdir(source_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
random.seed(42) # Keeps the split the same every time you run it
random.shuffle(all_images)

# split
split_index = int(len(all_images) * 0.70)
train_list = all_images[:split_index]
val_list = all_images[split_index:]

print(f"Total images: {len(all_images)}")
print(f"Copying {len(train_list)} to Train and {len(val_list)} to Val...")

# copying function
def copy_data(file_list, dest_img, dest_mask):
    copied_count = 0
    for filename in file_list:
        # image paths
        src_img = os.path.join(source_img_dir, filename)
        dst_img = os.path.join(dest_img, filename)
        
        # mask paths 
        src_mask = os.path.join(source_mask_dir, filename)
        dst_mask = os.path.join(dest_mask, filename)

        # copy Image
        shutil.copy(src_img, dst_img)
        
        # copy Mask if it exists
        if os.path.exists(src_mask):
            shutil.copy(src_mask, dst_mask)
        
        copied_count += 1
    return copied_count

# execute the copies
train_count = copy_data(train_list, train_img_out, train_mask_out)
val_count = copy_data(val_list, val_img_out, val_mask_out)

print("-" * 30)
print(f"Train: {train_count} pairs | Val: {val_count} pairs")