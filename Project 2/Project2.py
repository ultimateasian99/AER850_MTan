# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# tensorflow 2.18 Keras 3.6
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense 
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from tensorflow import keras
from keras import Sequential
import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau

#STEP 1 DATA PROCESSING 

# image dimensions
img_height = 500 
img_width =  500
batchsize = 32

# Relative directory paths
train_data_dir = 'Data/train'
valid_data_dir = 'Data/valid'

# data augmentation for training and validation
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height,img_width)
else:
    input_shape = (img_height,img_width,3)


# data generator and augmentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2, 
    zoom_range = 0.2
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batchsize,
    shuffle = True,
    class_mode = 'categorical'
)

valid_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2, 
    zoom_range = 0.2
)

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size = (img_height, img_width),
    batch_size = batchsize,
    shuffle = True,
    class_mode = 'categorical'
)
#STEP 2 NEURAL NETWORK ARCHITECTURE DESIGN

#define sequential model 
model = Sequential()

#feature learning layers of NN architecure
#model.add(Input(shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))  
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#classification layers of NN architecture
model.add(Flatten())                              # convert to 1d vector

model.add(Dense(32,activation='relu',kernel_regularizer=l2(0.0001)))           
model.add(Dropout(0.3))
model.add(Dense(3,activation='softmax'))


#compile model
model.compile(
    optimizer = Adam(0.001),          
    loss = 'categorical_crossentropy',             
    metrics = ['accuracy']
)
print(model.summary())

#training mi model
history = model.fit(
    train_generator,
    epochs = 10,                    #originally 50
    validation_data=valid_generator
)

#save the entire model as a .keras zip archive
model.save('mymodel.keras')


#Plotting time
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epoch_range = range(len(train_acc))

#STEP 4 MODEL EVALUATION
plt.figure()
plt.subplot(1,2,1)
plt.plot(epoch_range,train_acc, label = "Training Accuracy")
plt.plot(epoch_range,val_acc, label = "Validation Accuracy")
plt.legend(loc = "lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1,2,2)
plt.plot(epoch_range,train_loss, label = "Training Loss")
plt.plot(epoch_range,val_loss, label = "Validation Loss")
plt.legend(loc = "upper right")
plt.title("Training and Validation Loss")

plt.show()



