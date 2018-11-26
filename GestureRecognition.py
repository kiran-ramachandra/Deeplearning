# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:56:35 2018

@author: Kiran Ramachandra
"""

import xlrd
import numpy as np
import tensorflow as tf
import os

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.relu, kernel_initializer = 'random_normal', bias_initializer = 'zeros'))
    model.add(tf.keras.layers.Dense(10,activation=tf.nn.relu, kernel_initializer = 'random_normal', bias_initializer = 'zeros'))
    model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax, kernel_initializer = 'random_normal', bias_initializer = 'zeros'))
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
    return model

workBook = xlrd.open_workbook("FeaturesSet.xlsx")
sheet1 = workBook.sheet_by_index(0)
sheet2 = workBook.sheet_by_index(1)

checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

xTrain = np.empty([282,135],dtype = float)
yTrain = np.empty([282,1])
xTest = np.empty([1,135],dtype = float)
yTest = np.empty([1])

for i in range(282):
    xTrain[i,:] = np.array([value for value in sheet1.col_values(i)])
    
xTest = np.array([value for value in sheet2.col_values(15)])

xTest = xTest.reshape([1,135])
classes = [0, 1]

for i in range(0,140):
	yTrain[i,0] = classes[0]    #Up

for i in range(140,282):
	yTrain[i,0] = classes[1]   #Down
    
# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=3)

model = create_model()
#for the initial setup - train the model and store the weights in checkpoints
model.fit(xTrain, yTrain, epochs = 3, callbacks = [cp_callback], validation_data = (xTest,yTest),verbose=0)
#Later load the weights from the checkpoints - here the previous line of model.fit is to be commented
model.load_weights(checkpoint_path)
val_loss, val_acc = model.evaluate(xTest, yTest)

print('val_loss' = val_loss, 'val_acc' = val_acc)

#predict the first test data
predictions = model.predict(xTest)
print("prediction: " + str(predictions))

if(predictions[0,0] >= predictions[0,1]):
    print("Up Gesture")
    
if(predictions[0,0] < predictions[0,1]):
    print("Down Gesture")
