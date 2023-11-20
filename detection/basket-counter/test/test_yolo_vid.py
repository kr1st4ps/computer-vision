from ultralytics import YOLO
import sys

model = YOLO("models/best.pt")

results = model.track(source=sys.argv[1], show=True, imgsz=1008, classes=[0,1,3,4])