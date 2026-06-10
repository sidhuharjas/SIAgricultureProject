import os
import cv2
import numpy as np
import glob

DATASET_DIR = "/users/PZS1154/harjassidhu/SIAgricultureProject/ready_for_annotation"

CLASS_ID = 0  # main plant only

def process_and_annotate(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green range tuned for your plant
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False

    # Pick the largest contour = main plant
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # Ignore tiny detections
    if area < 1500:
        return False

    x, y, bw, bh = cv2.boundingRect(cnt)

    # Normalize for YOLO
    x_center = (x + bw/2) / w
    y_center = (y + bh/2) / h
    nw = bw / w
    nh = bh / h

    yolo_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}"

    label_path = os.path.splitext(img_path)[0] + ".txt"
    with open(label_path, "w") as f:
        f.write(yolo_line + "\n")

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

    print("\nDONE — Clean single-box labels generated.")
