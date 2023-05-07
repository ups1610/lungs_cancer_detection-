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

class visualize:
    def __init__(self,path):
        self.path=path

    def make_list(self):
        self.classes = os.listdir(self.path) 
        return self.classes   

    def show_img(self):
        for cat in self.classes:
           self.image_dir = f'{self.path}/{cat}'
           images = os.listdir(self.image_dir)
    
           fig, ax =plt.subplots(1,3,figsize=(15,5))
           fig.suptitle(f'Images for {cat} category....', fontsize = 20)
    
           for i in range(3):
               k = np.random.randint(0,len(images))
               img = np.array(Image.open(f'{self.path}/{cat}/{images[k]}'))
               ax[i].imshow(img)
               ax[i].axis('off')
               plt.show()    