import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import sys
import os
import shutil
import cv2
from tensorflow import keras

import utils.classification as cl

#   Collects arguments that were passed
source_dir = sys.argv[1]
target_dir = sys.argv[2]

#   Loads in the classification model
model = tf.keras.models.load_model("classification/finger_count.h5")

#   Define class names of the model
class_names = ['FIVE', 'FOUR', 'NONE', 'ONE', 'THREE', 'TWO']

#   Iterate through the files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):

        #   Read in and process the image
        img = cv2.imread(source_dir + "/" + filename)
        img = cl.process_img(img)

        #   Make prediction        
        pred = class_names[np.argmax(model.predict(img)[0])]

        #   Copy file from source dir to target dir
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, pred, filename)
        shutil.copy(source_path, target_path)