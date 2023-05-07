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

import warnings
warnings.filterwarnings('ignore')

from zipfile import ZipFile

class dataset:
   def __init__(self,data_path):
      self.data_path = data_path
      
   def get_data(self):
      with ZipFile(self.data_path,'r') as zip:
         zip.extractall()
         print('The data set has been extracted.')


if __name__ == "__main__":
   data=dataset('lungs_data.zip')
   print(data.get_data())

