from ultralytics import YOLO
import sys

yolo = YOLO('yolov8n.pt')
yolo.train(data=sys.argv[1], epochs=5)
valid_results = yolo.val()
print(valid_results)