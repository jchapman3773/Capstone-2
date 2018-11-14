# df = pd.read_csv('train-annotations-human-imagelabels.csv')
# df has ImageID,Source,LabelName,Confidence

from __future__ import print_function
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import keras

batch_size = 5
nb_classes = 4
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows,img_cols,3)
# number of convolutional filters to use
nb_filters = 40
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

train_data_dir = '../data/Banana_People_Not/train'
validation_data_dir = '../data/Banana_People_Not/validation'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

model = Sequential()
model.add(Convolution2D(nb_filters,
                        kernel_size,
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters,
                        kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.1))
# transition to an mlp
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

mc = keras.callbacks.ModelCheckpoint('Simple_CNN_Model_BPN.hdf5',
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)

model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=nb_epoch,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[mc])

# score = model.evaluate_generator(validation_generator, steps=len(validation_generator), verbose=0)
# print(score)
