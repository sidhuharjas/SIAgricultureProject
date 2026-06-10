import os
import glob

LABEL_DIR = "/users/PZS1154/harjassidhu/SIAgricultureProject/ready_for_annotation"

# thresholds you can tune
MIN_W = 0.02
MIN_H = 0.02
MAX_W = 0.80
MAX_H = 0.80
EDGE_MARGIN = 0.02
MAX_ASPECT_RATIO = 6.0   # too long = weed stem false positive

def boxes_overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax1, ay1 = ax - aw/2, ay - ah/2
    ax2, ay2 = ax + aw/2, ay + ah/2

    bx1, by1 = bx - bw/2, by - bh/2
    bx2, by2 = bx + bw/2, by + bh/2

    return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > bx2)

def merge_boxes(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax1, ay1 = ax - aw/2, ay - ah/2
    ax2, ay2 = ax + aw/2, ay + ah/2

    bx1, by1 = bx - bw/2, by - bh/2
    bx2, by2 = bx + bw/2, by + bh/2

    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    x2 = max(ax2, bx2)
    y2 = max(ay2, by2)

    new_w = x2 - x1
    new_h = y2 - y1
    new_x = x1 + new_w/2
    new_y = y1 + new_h/2

    return (new_x, new_y, new_w, new_h)

def clean_labels(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()

    cleaned = []
    boxes = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:5])

        # remove tiny boxes
        if w < MIN_W or h < MIN_H:
            continue

        # remove huge boxes
        if w > MAX_W or h > MAX_H:
            continue

        # remove boxes touching edges
        if x < EDGE_MARGIN or x > 1 - EDGE_MARGIN:
            continue
        if y < EDGE_MARGIN or y > 1 - EDGE_MARGIN:
            continue

        # remove weird aspect ratios
        aspect = max(w/h, h/w)
        if aspect > MAX_ASPECT_RATIO:
            continue

        boxes.append((cls, x, y, w, h))

    # merge overlapping boxes
    merged = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue

        cls_i, x_i, y_i, w_i, h_i = boxes[i]
        merged_box = (x_i, y_i, w_i, h_i)

        for j in range(i+1, len(boxes)):
            if j in used:
                continue

            cls_j, x_j, y_j, w_j, h_j = boxes[j]

            if boxes_overlap(merged_box, (x_j, y_j, w_j, h_j)):
                merged_box = merge_boxes(merged_box, (x_j, y_j, w_j, h_j))
                used.add(j)

        merged.append((cls_i, *merged_box))

    # write cleaned labels
    with open(label_path, "w") as f:
        for cls, x, y, w, h in merged:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def run_cleanup():
    label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    print(f"Found {len(label_files)} label files to clean.")

    for i, label_path in enumerate(label_files):
        clean_labels(label_path)

        if (i+1) % 200 == 0:
            print(f"Cleaned {i+1}/{len(label_files)} files...")

    print("Cleanup complete.")

if __name__ == "__main__":
    run_cleanup()
