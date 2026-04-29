
import os

from ultralytics import YOLO


#Load the model

model = YOLO("yolo11s-seg.pt")

#training
results = model.train(
    data="/home/mayurf/main_tasks/kiros/kiros/synthetic_data/heavy_guy/dataset_split/dataset_yolo_style/dataset.yaml",    
    epochs=80,              
    optimizer='AdamW',                
    lr0=0.01,                
    lrf=0.01,                # final learning rate fraction (cosine decay)
    weight_decay=0.0005,     # regularization to prevent overfitting
    warmup_epochs=3.0,
    device=0,                #  GPU
    project="heavy_guy_seg", 
    name="run_1",            
    save=True,               # save checkpoints
    exist_ok=True            # overwrite if run_1 already exists
)

# validate the model intemrinal
metrics = model.val()
print(f"Mean Average Precision (Mask): {metrics.seg.map}") 




