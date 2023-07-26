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
import pydot

import tensorflow as tf
from tensorflow import keras
from keras import layers

from data_visualization import visualize
from pre_processing import preprocess
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

class build_model:
    def __init__(self):
        pre = preprocess('lung_colon_image_set/lung_image_sets')
        self.img_size,self.split,self.epochs,self.batch_size = pre.hyp_tune()
        self.X, self.one_hot_encode = pre.convert(256)
        
    def training(self):
        self.X_train , self.X_test , self.Y_train, self.Y_test = train_test_split(self.X, self.one_hot_encode , test_size=0.2,random_state=2022)
        return self.X_train.shape, self.X_test.shape 
    
    def classifer(self,img_size):
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
        keras.utils.plot_model(
            model,
            show_shapes = True,
            show_dtype = True,
            show_layer_activations = True
        )
        
        return model
    
    def compilation(self,model):
        model.compile(
                      optimizer = 'adam',
                      loss = 'sparse_categorical_crossentropy',
                      metrics = ['accuracy']
                     )
        class mycallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get('val_accuracy') > 0.90:
                    print('\n Validation accuracy has reached upto \90% so, stopping further training.')
                    self.model.stop_training = True
        es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor='val_loss',patience=2,factor=0.5,verbose=1)
        print(model)
        history = model.fit(self.X_train, self.Y_train,
                    validation_data = (self.X_test, self.Y_test),
                    batch_size = self.batch_size, 
                    epochs = self.epochs, 
                    verbose = 1,
                    callbacks = [es , lr , mycallback()]
                    )
        return history
        
    def visualize_accuracy(self,history):
        history_df = pd.DataFrame(history.history)
        history_df.loc[:,['loss','val_loss']].plot()
        history_df.loc[:,['accuracy','val_accuracy']].plot()
        plt.show()  

    def model_evaluation(self,model):
        Y_pred = model.predict(self.X_test)
        Y_val = np.argmax(self.Y_test, axis=1)
        Y_pred = np.argmax(Y_pred, axis=1)  
        return Y_val,Y_pred

    def confusion_matrix(self,Y_val, Y_pred):
        metrics.confusion_matrix(Y_val, Y_pred) 
        vi = visualize('lung_colon_image_set/lung_image_sets') 
        classes = vi.make_list()
        print(metrics.classification_report(Y_val, Y_pred,target_names=classes)) 


if __name__ == "__main__":
    m_obj = build_model()
    print(m_obj.training())
    model = m_obj.classifer(256)
    history = m_obj.compilation(model)
    m_obj.visualize_accuracy(history)
    Y_val, Y_pred = m_obj.model_evaluation(model)
    m_obj.confusion_matrix(Y_val , Y_pred)
    
    

