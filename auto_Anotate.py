import os
import cv2
import numpy as np
import glob

# ==============================================================================
# CONFIG
# ==============================================================================
DATASET_DIR = "/users/PZS1154/harjassidhu/SIAgricultureProject/ready_for_annotation"

CROP_CLASS_ID = 0
WEED_CLASS_ID = 1

# ==============================================================================
# DETECTION + LABELING
# ==============================================================================
def process_and_annotate(img_path):
    """Detects vegetation using Excess Green and writes YOLO labels next to the image."""
    
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]

    # Excess Green Index (simple but effective for plants)
    b, g, r = cv2.split(img.astype(np.float32))
    exg = (2 * g - r - b).clip(0, 255).astype(np.uint8)

    # Otsu threshold to isolate green regions
    _, mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove tiny specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Find blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Simple size-based classification
        class_id = CROP_CLASS_ID if area > 4500 else WEED_CLASS_ID

        # Normalize for YOLO
        x_center = (x + bw / 2) / w
        y_center = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

    # Save label file next to the image
    label_path = os.path.splitext(img_path)[0] + ".txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")

    return True

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"Scanning: {DATASET_DIR}")

    # Collect all images in the folder
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(DATASET_DIR, ext)))

    total = len(images)
    print(f"Found {total} images to annotate.")

    done = 0
    for i, img_path in enumerate(images):
        if process_and_annotate(img_path):
            done += 1

        # Progress update
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} images...")

    print("\n" + "=" * 60)
    print("AUTO‑ANNOTATION COMPLETE")
    print(f"Labeled {done} images.")
    print(f"Labels saved in: {DATASET_DIR}")
    print("=" * 60)
