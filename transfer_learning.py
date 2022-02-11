# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:56:31 2021

@author: DELL
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#load the pretrained model
pretrained_model = tf.keras.models.load_model('full_mnist_model.h5')
pretrained_model.layers


for layer in pretrained_model.layers:
    print(f'{layer}:',layer.trainable)
    
#make trainable false to all layers except last layer

for layer in pretrained_model.layers[:-1]:
    layer.trainable = False
    print(f'{layer}:',layer.trainable)
    
layers = pretrained_model.layers[:-1]    
      
new_model = tf.keras.models.Sequential(layers)
new_model.add(tf.keras.layers.Dense(2,activation = 'softmax'))

new_model.summary()

#now we have the model skelton
#will trian the same mnist dataset, but this time will predict either input digit is even or odd
#for this will need to change the labels, becuase lable has 0 to 9
# we need two labels only either even or odd

#load the dataset
(X_train_full,y_train_full),(X_test,y_test) = tf.keras.datasets.mnist.load_data()
X_train_full = X_train_full/255
X_test = X_test/255

X_valid,X_train = X_train_full[:5000],X_train_full[5000:]
y_valid,y_train = y_train_full[:5000],y_train_full[5000:]

#converting y_train,y_test and y_valid into two classes
#current situation

y_train.nunique

np.unique(y_train)

def convert_label(label):
    for id,data in enumerate(label):
        if data%2 == 0:
            label[id] = 1
        else:
            label[id] = 0
            
convert_label(y_train) 

convert_label(y_test)  

convert_label(y_valid)     


#now we have data
np.unique(y_train)


#compile and fit the model

new_model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3),
              metrics = ['accuracy'])


history =new_model.fit(X_train,y_train,epochs = 10,validation_data = (X_valid,y_valid),verbose = 2)

new_model.evaluate(X_test,y_test)

new_model.summary()












    
    