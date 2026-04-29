import os
from PIL import Image
from tqdm import tqdm

# configure
src_dir = r"/home/mayurf/main_tasks/kiros/kiros/synthetic_data/heavy_guy/dataset_split/dataset_yolo_style/val/images_old"
dst_dir = r"/home/mayurf/main_tasks/kiros/kiros/synthetic_data/heavy_guy/dataset_split/dataset_yolo_style/val/images"
scale_factor = 0.3  # shrink to 30% of original size
quality = 85         # high quality compression for JPEG

# check if  destination exists
os.makedirs(dst_dir, exist_ok=True)

# get list of images
extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
files = [f for f in os.listdir(src_dir) if f.lower().endswith(extensions)]

print(f"Found {len(files)} images. Resizing to {scale_factor*100}%...")

for filename in tqdm(files):
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
    
    try:
        with Image.open(src_path) as img:
            # 1. Calculate new dimensions (30% of original)
            new_w = int(img.width * scale_factor)
            new_h = int(img.height * scale_factor)
            
            # 2. Resize
            # Using Image.LANCZOS to keep the cube edges crisp for the AI
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            
            # 3. Save
            if filename.lower().endswith('.png'):
                img_resized.save(dst_path, optimize=True)
            else:
                img_resized.save(dst_path, quality=quality, optimize=True)
                
    except Exception as e:
        print(f"error processing {filename}: {e}")

print(f"\ndone! saved to: {dst_dir}")


