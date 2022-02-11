# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:15:44 2021

@author: DELL
"""

import numpy as np
import pandas as pd
import tensorflow as tf

#load the data
(X_train_full,y_train_full),(X_test,y_test) = tf.keras.datasets.mnist.load_data()

X_train_full = X_train_full/255
X_test = X_test/255

X_valid,X_train = X_train_full[:5000],X_train_full[5000:]
y_valid,y_train = y_train_full[:5000],y_train_full[5000:]


layers = [
            tf.keras.layers.Flatten(input_shape = [28,28]),
            tf.keras.layers.Dense(300,activation = 'LeakyReLU',kernel_initializer = "he_normal"),
            tf.keras.layers.Dense(100,activation = 'LeakyReLU',kernel_initializer = "he_normal"),
            tf.keras.layers.Dense(10,activation = 'softmax',kernel_initializer = "he_normal")
            
            ]    

model = tf.keras.models.Sequential(layers)
                              

model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3),
              metrics = ['accuracy'])

model.summary()

history = model.fit(X_train,y_train,epochs = 10,validation_data = (X_valid,y_valid),verbose = 2)

model.save("full_mnist_model.h5")

###########################################################################
















