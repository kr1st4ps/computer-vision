from ultralytics import YOLO
import cv2
import sys

model = YOLO("models/best.pt")

img = cv2.imread(sys.argv[1])

results = model.predict(img, stream=True)                      # run prediction on img
for result in results:                                         # iterate results
    boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
    for box in boxes:                                          # iterate boxes
        r = box.xyxy[0].astype(int)                            # get corner points as int
        print(r)                                               # print boxes
        cv2.rectangle(img, r[:2], r[2:], (255, 255, 255), 2) 

cv2.imshow('My Image Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()