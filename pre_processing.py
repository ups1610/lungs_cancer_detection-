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

from data_visualization import visualize

class preprocess:
    def __init__(self,path):
        vi = visualize('lung_colon_image_set/lung_image_sets')
        self.classes = vi.make_list()
        self.path = path

    def hyp_tune(self):
        self.img_size = 256
        self.split = 0.2 
        self.epochs = 10 
        self.batch_size = 64
        return self.img_size, self.split , self.epochs , self.batch_size

    def convert(self,img_size):
        X=[]
        Y=[]
        for i, cat in enumerate(self.classes):
            images = glob(f'{self.path}/{cat}/*.jpeg')

            for image in images:
                img = cv2.imread(image)

                X.append(cv2.resize(img, (img_size, img_size)))
                Y.append(i)
            X = np.asarray(X)
            one_hot_encoded_Y = pd.get_dummies(Y).values
            return X,one_hot_encoded_Y   


if __name__ == "__main__":
    p = preprocess('lung_colon_image_set/lung_image_sets')  
    print(p.hyp_tune(256,0.2,10,64)) 
    print(p.convert(256)) 



