import os
from ultralytics import YOLO

def main():
    # 1. Find your best-trained weights folder
    # Change 'train-15' to your actual latest folder name that completes successfully
    model_path = "D:/SI/runs/segment/train-15/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Cannot find weights at {model_path} to compress. Finish training first!")
        return

    print("Loading your custom trained weights...")
    model = YOLO(model_path)

    print("Compressing and optimizing model to be ultra-lightweight...")
    
    # 2. Export the model to ONNX format with INT8 quantization (half-precision math)
    # This reduces file size drastically and speeds up CPU/GPU processing
    exported_path = model.export(
        format="onnx", 
        half=True,       # Uses FP16 half-precision numbers to cut the file size in half
        simplify=True    # Fuses neural network layers together so it runs faster
    )
    
    print("\n" + "="*50)
    print("LIGHTWEIGHT MODEL READY!")
    print(f"Your hyper-fast model is saved as: {exported_path}")
    print("Use this new '.onnx' file in your video analysis script for pure speed!")
    print("="*50)

if __name__ == "__main__":
    main()