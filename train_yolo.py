# train_yolo.py
from ultralytics import YOLO

def main():
    # load a pretrained YOLOv8 model (nano = fast, good start)
    model = YOLO("yolov8n.pt")

    model.train(
        data="/users/PZS1154/harjassidhu/SIAgricultureProject/dataset/data.yaml",
        epochs=200,          # crank this up on supercomputer
        imgsz=640,
        batch=8,
        workers=8,
        project="runs_yolo",
        name="plant_detector",
    )

if __name__ == "__main__":
    main()
