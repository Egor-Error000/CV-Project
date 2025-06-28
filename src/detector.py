from ultralytics import YOLO
import cv2
import torch

def load_model(weights_path="yolov8x.pt"):
    return YOLO(weights_path)

def detect_people(model, frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb, imgsz=1280, conf=0.3, iou=0.6,
                            device='cuda' if torch.cuda.is_available() else 'cpu')[0]

    detections = []
    for box, cls_id, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                  results.boxes.cls.cpu().numpy().astype(int),
                                  results.boxes.conf.cpu().numpy()):
        if cls_id == 0:  # Только люди
            x1, y1, x2, y2 = map(int, box)
            detections.append([[x1, y1, x2, y2], float(conf), cls_id])
    return detections
