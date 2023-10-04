import cv2
import numpy as np

#   Function that takes an image and proceses it to binary format and turns into an array to be used for classifiaction
def process_img(img):
    img = cv2.resize(img, (100,100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7,7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, roi = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = np.float32(roi)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    return img