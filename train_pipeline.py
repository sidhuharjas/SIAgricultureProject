import torch
from ultralytics import YOLO

def main():
    # 1. Load a fresh, blank YOLOv8 segmentation model
    model = YOLO("yolov8n-seg.pt")

    print("Starting training with data augmentations...")

    # 2. Determine hardware dynamically (Uses GPU/CUDA if available, otherwise CPU)
    if torch.cuda.is_available():
        device_target = "cuda"
        print("Hardware Target: GPU (CUDA)")
    else:
        device_target = "cpu"
        print("Hardware Target: CPU (Note: Training will be slower on CPU)")

    # 3. Train the model with built-in data distortion settings
    model.train(
        data="dataset.yaml",               # Points to the local config file in your directory
        epochs=50,                         # Total training cycles
        imgsz=640,                         # Standard image resolution size
        device=device_target,              # Uses the dynamically selected device
        
        # --- DATA AUGMENTATION HYPERPARAMETERS ("Make it weird") ---
        degrees=45.0,      # Randomly rotate images up to 45 degrees
        shear=15.0,        # Randomly tilt images up to 15 degrees
        perspective=0.001, # Adds slight 3D perspective warp
        flipud=0.5,        # 50% chance to flip images upside down
        fliplr=0.5,        # 50% chance to flip images left-to-right
        hsv_h=0.015,       # Random color hue adjustments
        hsv_s=0.7,         # Random color saturation changes
        hsv_v=0.4,         # Random brightness changes
        scale=0.5,         # Randomly zoom in or out on the vegetation
        mosaic=1.0         # Combines 4 random training images into a single training collage
    )
    
    print("Training process configuration complete.")

if __name__ == "__main__":
    main()