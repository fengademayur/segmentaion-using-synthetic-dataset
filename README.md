This project demonstrates a complete end-to-end pipeline for object segmentation without manual data labeling. 
Using Blender as a synthetic data engine, I generated a custom dataset for a "Heavy Guy" class and trained a YOLOv11 segmentation model.

The Workflow

    Synthetic Generation: Used Blender Python API to render 3D characters with randomized lighting and camera angles.

    Automated Masking: Generated pixel-perfect segmentation masks directly from the 3D engine, eliminating human labeling error.

    Background Merging: Augmented the synthetic subjects with real-world backgrounds to bridge the gap between simulation and reality.

    Training: Trained on a local NVIDIA RTX 4060 using the Ultralytics framework.

    Live Inference: Deployed the model on an Intel RealSense D435 for real-time segmentation.

Tech Stack

    OS: Linux

    GPU: NVIDIA GeForce RTX 4060 (8GB VRAM)

    Software: Blender 4.x, Python 3.x, OpenCV

    AI Framework: Ultralytics YOLOv11-seg

    Camera: Intel RealSense D435


    Result: 

    The model was trained for 100 epochs, reaching stable convergence. Despite being trained on 100% synthetic characters, the model generalizes effectively to real-world footage captured by the depth camera.
    
