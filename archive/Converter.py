import ultralytics
from ultralytics import YOLO
model = YOLO('custom_yolov8.pt')
model.export(format='onnx')
