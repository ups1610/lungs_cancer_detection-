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

from pre_processing import preprocess as pre

# class build_model:
#     def __init__(self):
#         self.path = 