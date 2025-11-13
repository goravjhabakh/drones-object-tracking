from ultralytics import YOLO
from core.config import YOLO_MODEL_PATH

yolo_model = YOLO(YOLO_MODEL_PATH)

def detect_objects(frame):
  results = yolo_model(frame)
  return results