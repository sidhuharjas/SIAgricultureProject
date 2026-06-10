import os
import time
import random
import glob
from PIL import Image, ImageEnhance

# ==============================================================================
# 1. CONFIGURATION & RANDOMNESS INITIALIZATION
# ==============================================================================
# High‑resolution dynamic seed for true randomness
dynamic_seed = int(time.time() * 1_000_000) & 0xFFFFFFFF
random.seed(dynamic_seed)

BASE_DIR = "/users/PZS1154/harjassidhu/SIAgricultureProject"
OUTPUT_DIR = os.path.join(BASE_DIR, "ready_for_annotation")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INIT] Output directory ready: {OUTPUT_DIR}")

# ==============================================================================
# 2. AUGMENTATION HELPERS
# ==============================================================================
def apply_random_augmentation(img_path):
    """
    Applies random brightness, contrast, saturation, and optional horizontal flip.
    Returns the augmented PIL image and a boolean indicating if a flip occurred.
    """
    img = Image.open(img_path).convert("RGB")
    flipped = False

    # Random brightness (0.7–1.3)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

    # Random contrast (0.7–1.3)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

    # Random saturation (0.6–1.4)
    img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))

    # 50% chance horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped = True

    return img, flipped


def process_yolo_label(src_label_path, dst_label_path, flipped):
    """
    Copies YOLO label file and adjusts x-center if the image was flipped.
    """
    if not os.path.exists(src_label_path):
        # Create empty label file if none exists
        open(dst_label_path, "w").close()
        return

    with open(src_label_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            new_lines.append(line.strip())
            continue

        class_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Flip x-center if image was flipped
        if flipped:
            x_center = 1.0 - x_center

        extra = " ".join(parts[5:])
        formatted = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        if extra:
            formatted += f" {extra}"

        new_lines.append(formatted)

    with open(dst_label_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

# ==============================================================================
# 3. PIPELINE EXECUTION
# ==============================================================================
def run_pipeline():
    # Collect all images except those already in the output directory
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    raw_images = []

    for ext in extensions:
        for f in glob.glob(os.path.join(BASE_DIR, "**", ext), recursive=True):
            if "ready_for_annotation" not in f:
                raw_images.append(f)

    raw_images = list(set(raw_images))
    total_raw = len(raw_images)

    print(f"[SCAN] Found {total_raw} raw images.")
    print("[RUN] Generating 3 random augmentations per image...")

    total_generated = 0

    for img_path in raw_images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        src_label_path = os.path.splitext(img_path)[0] + ".txt"

        for v in range(1, 4):
            dst_img = os.path.join(OUTPUT_DIR, f"{base_name}_aug_v{v}.jpg")
            dst_lbl = os.path.join(OUTPUT_DIR, f"{base_name}_aug_v{v}.txt")

            aug_img, flipped = apply_random_augmentation(img_path)
            aug_img.save(dst_img, quality=95)

            process_yolo_label(src_label_path, dst_lbl, flipped)
            total_generated += 1

    print("\n" + "=" * 55)
    print("[DONE] DATA AUGMENTATION COMPLETE")
    print(f"Generated: {total_generated} images + {total_generated} labels")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()
