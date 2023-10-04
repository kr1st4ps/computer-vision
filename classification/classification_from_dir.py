import numpy as np
import tensorflow as tf
import sys
import os
import shutil
import cv2
from multiprocessing import Pool as ProcessPool

import utils.classification as cl

def predict_img(filename, model, source_dir, target_dir):
    #   Read in and process the image
    img = cv2.imread(source_dir + "/" + filename)
    img = cl.process_img(img)

    #   Make prediction        
    pred = class_names[np.argmax(model.predict(img)[0])]

    #   Copy file from source dir to target dir
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, pred, filename)
    shutil.copy(source_path, target_path)

#   Collects arguments that were passed
source_dir = sys.argv[1]
target_dir = sys.argv[2]

#   Loads in the classification model
model = tf.keras.models.load_model("classification/finger_count.h5")

#   Define class names of the model
class_names = ['FIVE', 'FOUR', 'NONE', 'ONE', 'THREE', 'TWO']

if __name__ == "__main__":
    desired_file_list = [file_name for file_name in os.listdir("/Users/kristapsalmanis/Downloads/data") if file_name.endswith(".png")]

    with ProcessPool(processes=4) as pool:
        results = pool.map(predict_img, desired_file_list)