cat << 'EOF' > prep_pipeline.py
import os
import cv2
import random
import numpy as np
from pathlib import Path

def create_output_structure(base_dir):
    """Creates the target directory structure for annotation."""
    output_path = Path(base_dir) / "ready_for_annotation"
    images_path = output_path / "images"
    labels_path = output_path / "labels"
    
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    return images_path, labels_path

def apply_weird_augmentations(image):
    """Applies random rotations, color distortions, and tilts."""
    h, w = image.shape[:2]
    
    # 1. Random Rotation (between -45 and 45 degrees)
    angle = random.uniform(-45, 45)
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # 2. Random HSV Color Distortion (Brightness/Saturation)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.6, 1.4) # Saturation
    hsv[:, :, 2] *= random.uniform(0.6, 1.4) # Brightness
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 3. Random Horizontal Flip (50% chance)
    if random.choice([True, False]):
        image = cv2.flip(image, 1)
        
    return image

def main():
    # Start tracking directly from the main symlink folder
    source_dir = Path("crop_weed_data")
    project_dir = Path("/users/PZS1154/harjassidhu/SIAgricultureProject")
    
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} not found.")
        return

    # Set up destination folders
    img_out, lbl_out = create_output_structure(project_dir)
    print(f"Target directories initialized at: {project_dir / 'ready_for_annotation'}")

    # Use rglob to scan recursively through all subdirectories (like agri_data/data)
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.PNG", "*.JPG", "*.JPEG"]
    image_files = []
    for ext in supported_extensions:
        image_files.extend(list(source_dir.rglob(ext)))

    if not image_files:
        print(f"Error: Still couldn't find any images inside {source_dir} using recursive search.")
        return

    print(f"Found {len(image_files)} raw images deep inside the directory tree.")
    print("Generating augmented variations...")

    counter = 0
    for img_path in image_files:
        raw_img = cv2.imread(str(img_path))
        if raw_img is None:
            continue
            
        base_name = img_path.stem
        
        # Generate 3 variations per image
        for i in range(3):
            weird_img = apply_weird_augmentations(raw_img)
            
            # Save format setup
            new_filename = f"aug_{base_name}_{i}.jpg"
            target_image_path = img_out / new_filename
            cv2.imwrite(str(target_image_path), weird_img)
            
            # Create matching empty label file
            target_label_path = lbl_out / f"aug_{base_name}_{i}.txt"
            open(target_label_path, 'a').close()
            
            counter += 1

    print("\n" + "="*50)
    print("PIPELINE COMPLETE: DATASET IS READY FOR ANNOTATION")
    print(f"Total files generated: {counter} images & {counter} matching label files.")
    print(f"Output Location: {project_dir}/ready_for_annotation/")
    print("="*50)

if __name__ == "__main__":
    main()
EOF
