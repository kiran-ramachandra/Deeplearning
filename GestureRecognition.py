# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:56:35 2018

@author: Kiram Ramachandra
"""

import xlrd
import numpy as np
import tensorflow as tf

workbook = xlrd.open_workbook("arrays.xlsx")
sheet = workbook.sheet_by_index(0)

x_train = np.empty([301,135],dtype = float)
y_train = np.empty([301,1])
x_test = np.empty([1,135],dtype = float)
y_test = np.empty([1])

for i in range(301):
    x_train[i,:] = np.array([value for value in sheet.col_values(i)])
    
x_test = np.array([value for value in sheet.col_values(301)])

x_test = x_test.reshape([1,135])
classes = [0, 1]

for i in range(0,151):
	y_train[i,0] = classes[0]

for i in range(151,301):
	y_train[i,0] = classes[1]
    


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))

model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy']
              )

model.fit(x_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(x_test, y_test)

#predict the first test data
predictions = model.predict(x_test)
print("prediction: " + str(predictions))
