import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import callbacks
np.random.seed(1337)  # for reproducibility

class SimpleCNN():

    def __init__(self,batch_size=8,nb_classes=4,nb_epoch=20,img_rows=128,img_cols=128,input_dim=3,
                                                nb_filters=64,pool_size=(2,2),kernel_size=(3,3)):
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.input_dim = input_dim
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.train_generator = None
        self.validation_generator = None
        self.holdout_generator = None
        self.model = None

    def make_generators(self,directory):
        train_data_dir = directory+'/train'
        validation_data_dir = directory+'/validation'
        holdout_data_dir = directory+'/holdout'

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=180,
            horizontal_flip=True,
            vertical_flip=True)

        test_datagen = ImageDataGenerator(
            rescale=1. / 255
            )

        self.train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.holdout_generator = test_datagen.flow_from_directory(
            holdout_data_dir,
            target_size=(self.img_rows, self.img_cols),
            batch_size=self.batch_size,
            class_mode='categorical')
        return self.train_generator, self.validation_generator, self.holdout_generator

    def make_model(self):
        self.model = Sequential()
        self.model.add(Convolution2D(self.nb_filters,
                                self.kernel_size,
                                input_shape=(self.img_rows,self.img_cols,self.input_dim)))
        self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(self.nb_filters,
        #                         self.kernel_size))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(self.nb_filters,
        #                         self.kernel_size))
        # self.model.add(MaxPooling2D(pool_size=self.pool_size))
        # self.model.add(Dropout(0.1))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(self.nb_filters,
        #                         self.kernel_size))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(self.nb_filters,
        #                         self.kernel_size))
        # self.model.add(MaxPooling2D(pool_size=self.pool_size))
        # self.model.add(Dropout(0.1))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(self.nb_filters,
        #                         self.kernel_size))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(self.nb_filters,
        #                         self.kernel_size))
        # self.model.add(MaxPooling2D(pool_size=self.pool_size))
        # self.model.add(Dropout(0.1))
        # self.model.add(Activation('relu'))
        # self.model.add(Convolution2D(self.nb_filters,
        #                         self.kernel_size))
        # self.model.add(Activation('relu'))
        self.model.add(Convolution2D(self.nb_filters,
                                self.kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Dropout(0.1))

        self.model.add(Convolution2D(self.nb_filters,
                                self.kernel_size))
        self.model.add(Activation('relu'))

        # transition to an mlp
        self.model.add(Flatten())
        # self.model.add(Dense(128))
        # self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.1))
        self.model.add(Dense(self.nb_classes))
        self.model.add(Activation('softmax'))

        adam = Adam(lr=0.001)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

    def fit(self):
        mc = callbacks.ModelCheckpoint('Simple_CNN_Model_BPN.h5',
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
        hist = callbacks.History()
        es = callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=0,
                                            verbose=0,
                                            mode='auto')
        if not os.path.exists('Simple_Cat_CNN_tensorboard'):
            os.makedirs('Simple_Cat_CNN_tensorboard')
        tensorboard = callbacks.TensorBoard(
                    log_dir='Simple_Cat_CNN_tensorboard',
                    histogram_freq=0,
                    batch_size=self.batch_size,
                    write_graph=True,
                    embeddings_freq=0)
        self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=self.nb_epoch,
            validation_data=self.validation_generator,
            validation_steps=len(self.validation_generator),
            callbacks=[mc,hist,es,tensorboard])

def confusion_matrix_CNN(predictions,true):
    pass

def open_saved_model(model_name,generator):
    model = load_model(model_name)
    evaluation = model.evaluate_generator(generator, steps=len(generator), verbose=0)
    print(evaluation)

if __name__ == '__main__':
    Banana_CNN = SimpleCNN()
    _, generator, _ = Banana_CNN.make_generators('../data/Banana_People_Not/4_Classes')
    Banana_CNN.make_model()
    Banana_CNN.fit()
    # open_saved_model('Simple_CNN_Model_BPN.h5',generator)
