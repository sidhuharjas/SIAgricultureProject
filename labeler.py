import os
import cv2
import numpy as np
import glob

DATASET_DIR = "/users/PZS1154/harjassidhu/SIAgricultureProject/ready_for_annotation"

CROP_CLASS_ID = 0
WEED_CLASS_ID = 1

def process_and_annotate(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]

    # Excess Green
    b, g, r = cv2.split(img.astype(np.float32))
    exg = (2*g - r - b).clip(0, 255).astype(np.uint8)

    # Adaptive threshold (better than Otsu)
    mask = cv2.adaptiveThreshold(
        exg, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        35, -5
    )

    # Morphological closing to merge leaves
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    # Opening to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:      # ignore tiny noise
            continue
        if area > 50000:    # ignore giant merged blobs
            continue

        # shape filtering
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.05:   # long skinny shapes = noise
            continue

        # bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)

        # classify by area
        class_id = CROP_CLASS_ID if area > 5000 else WEED_CLASS_ID

        # normalize
        x_center = (x + bw/2) / w
        y_center = (y + bh/2) / h
        nw = bw / w
        nh = bh / h

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")

    # If nothing detected, skip writing empty file
    if len(yolo_lines) == 0:
        return False

    label_path = os.path.splitext(img_path)[0] + ".txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")

    return True


if __name__ == "__main__":
    print(f"Scanning: {DATASET_DIR}")

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(DATASET_DIR, ext)))

    total = len(images)
    print(f"Found {total} images.")

    done = 0
    for i, img_path in enumerate(images):
        if process_and_annotate(img_path):
            done += 1

        if (i+1) % 200 == 0 or (i+1) == total:
            print(f"Processed {i+1}/{total}")

    print("\nDONE — Clean labels generated.")
