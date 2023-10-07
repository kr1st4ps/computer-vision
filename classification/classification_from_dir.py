import numpy as np
import tensorflow as tf
import sys
import os
import shutil
import cv2
from multiprocessing import Pool as ProcessPool

import utils.classification as cl

def predict_img(filename):
    #   Read in and process the image
    img = cv2.imread(filename)
    img = cl.process_img(img)

    #   Make prediction        
    pred = class_names[np.argmax(model.predict(img)[0])]

    #   Copy file from source dir to target dir
    target_path = os.path.join(target_dir, pred, filename.split("/")[-1])

    if move:
        shutil.move(filename, target_path)
    else:
        shutil.copy(filename, target_path)


move = False
paths = []
desired_file_list = []

#   Loads in the classification model
model = tf.keras.models.load_model("classification/finger_count.h5")

#   Define class names of the model
class_names = ['FIVE', 'FOUR', 'NONE', 'ONE', 'THREE', 'TWO']

#   Collects arguments that were passed
for arg in sys.argv[1:]:
    #   Move images instead of copy
    if arg == "-m":
        move = True
    
    #   Collect all passed paths
    else:
        paths.append(arg)

target_dir = paths.pop()
source_dirs = paths


if __name__ == "__main__":

    #   Collects all image paths
    for dir in source_dirs:
        for file_name in os.listdir(dir):
            if file_name.endswith(".png"):
                desired_file_list.append(dir + "/" + file_name)

    #   Starts classification process on multiple cores
    with ProcessPool(processes=4) as pool:
        results = pool.map(predict_img, desired_file_list)