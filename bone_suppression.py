# Importing Libraries
import cv2
# import glob
import os
# import time
from PIL import Image
import numpy as np
# import imageio
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from keras import preprocessing
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, Add, Input, Lambda
from tensorflow.keras.models import Model

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch

import pydicom
from tqdm import tqdm

# Defining the Residual Block
def res_block(x_in, filters, scaling):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x_in)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x

def resnet_bs(num_filters=64, num_res_blocks=16, res_block_scaling=None):
    x_in = Input(shape=(512, 512, 1)) # Input shape : To be changed as per the Input Image
    x = b = Conv2D(num_filters, (3, 3), padding='same')(x_in)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, (3, 3), padding='same')(b)
    x = Add()([x, b])
    x = Conv2D(1, (3, 3), padding='same')(x)
    return Model(x_in, x, name="ResNet-BS")

# Instantiating Model
model = resnet_bs(num_filters=64, num_res_blocks=22, res_block_scaling=0.1)

# Loading Weights
model.load_weights("./ResNet-BS.bestjsrt4500_num_filters_64_num_res_blocks_22.h5")
print("Wei et al. model loaded successfully!")

model.summary()

test_dir = "/Users/amograo/Desktop/dicom-images-test-stage_2"
output_dir = "/Users/amograo/Desktop/Bone_Suppressed_Test"
images=[f for f in os.listdir(test_dir) if f.endswith('.dcm')]
# Reading Images and Prediction
for image_name in tqdm(images):
    ds=pydicom.dcmread(os.path.join(test_dir,image_name))
    image_for_pred = Image.fromarray(ds.pixel_array)

    image_for_pred = image_for_pred.resize((512, 512))
    image_for_pred = img_to_array(image_for_pred)

    image_for_pred = image_for_pred.astype('float64') / 255.0
    image_for_pred = np.expand_dims(image_for_pred, axis=0)
    predict = model.predict(image_for_pred)
    predict = np.reshape(predict, (512, 512, 1))
    predict = predict * 255.0

    cv2.imwrite(os.path.join(output_dir, image_name.split('.')[0]+".png"), predict)