import os
import cv2
import numpy as np

#dataset path
BASE_PATH = r"/home/mayurf/main_tasks/kiros/kiros/synthetic_data/heavy_guy/dataset_split"

def mask_to_yolo_seg(mask_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segments = []
    for contour in contours:
        if len(contour) >= 6:  #YOLOv8 requires at least 6 points
            contour = contour.squeeze(1)
            norm_contour = contour / [w, h]  # normalize to 0–1
            segment = norm_contour.flatten().tolist()
            segments.append(segment)

    return segments

def generate_labels(mask_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    for filename in os.listdir(mask_dir):
        if filename.endswith('.png'):
            mask_path = os.path.join(mask_dir, filename)
            label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))

            segments = mask_to_yolo_seg(mask_path)
            with open(label_path, 'w') as f:
                for segment in segments:
                    f.write(f"0 " + " ".join([f"{p:.6f}" for p in segment]) + "\n")

# paths to train and val masks and labels
train_masks = os.path.join(BASE_PATH, 'train', 'masks')
train_labels = os.path.join(BASE_PATH, 'train', 'labels')
val_masks = os.path.join(BASE_PATH, 'val', 'masks')
val_labels = os.path.join(BASE_PATH, 'val', 'labels')

# run conversion
generate_labels(train_masks, train_labels)
generate_labels(val_masks, val_labels)

print("done")


