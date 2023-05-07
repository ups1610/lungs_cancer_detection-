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

class preprocess:
    def __init__(self,path):
        self.classes = vi.make_list()
        self.path = path

    def hyp_tune(self,img_size,split,epochs,batch_size):
        self.img_size = img_size
        self.split = split 
        self.epochs = epochs 
        self.batch_size = batch_size

    def convert(self):
        X=[]
        Y=[]
        for i, cat in enumerate(self.classes):
            images = glob(f'{self.path}/{cat}/*.jpeg')

            for image in images:
                img = cv2.imread(image)

                X.append(cv2.resize(img, (self.img_size, self.img_size)))
                Y.append(i)
            X = np.asarray(X)
            one_hot_encoded_Y = pd.get_dummies(Y).values
            return one_hot_encoded_Y     


