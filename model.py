#cnn model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn import metrics

import cv2
import gc
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers

from data_visualization import visualize as vi

from pre_processing import preprocess

class build_model:
    def __init__(self):
        pre = preprocess('lung_colon_image_set/lung_image_sets')
        self.X, self.one_hot_encode = pre.convert(256)
        
    def training(self):
        X_train , X_test , Y_train, Y_test = train_test_split(self.X, self.one_hot_encode , test_size=0.2,random_state=2022)
        return X_train.shape, X_test.shape 
    
    def apply_model(self,img_size):
        model = keras.models.Sequential([layers.Conv2D(
                                               filters =32,
                                               kernel_size=(5,5),
                                               activation='relu',
                                               input_shape=(img_size,img_size,3),
                                               padding='same'),
                layers.MaxPooling2D(2, 2),

                layers.Conv2D(filters=64,
                              kernel_size=(3, 3),
                              activation='relu',
                              padding='same'),
                layers.MaxPooling2D(2, 2),

                layers.Conv2D(filters=128,
                              kernel_size=(3, 3),
                              activation='relu',
                              padding='same'),
                layers.MaxPooling2D(2, 2),

                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.BatchNormalization(),
                layers.Dense(3, activation='softmax')
                ])
        return model
    
    def model_train(self,model):
        keras.utils.plot_model(
            model,
            show_shapes = True,
            show_dtype = True,
            show_layer_activations = True
        )


if __name__ == "__main__":
    mdl = build_model()
    print(mdl.training())
    get_model = mdl.apply_model(256)
    print(get_model.summary())
    mdl.model_train(get_model)
