import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import callbacks
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility

class SimpleCNN():

    def __init__(self,batch_size=8,nb_classes=4,nb_epoch=30,img_rows=300,img_cols=300,input_dim=3,
                                                nb_filters=128,pool_size=(2,2),kernel_size=(3,3),
                                                augmentation_strength=0.2):
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
        self.class_weights = None
        self.augmentation_strength = augmentation_strength

    def make_generators(self,directory):
        train_data_dir = directory+'/train'
        validation_data_dir = directory+'/validation'
        holdout_data_dir = directory+'/holdout'

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=15*self.augmentation_strength,
            width_shift_range=self.augmentation_strength,
            height_shift_range=self.augmentation_strength,
            shear_range=self.augmentation_strength,
            zoom_range=self.augmentation_strength,
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

    def _find_class_weights(self):
        counter = Counter(self.train_generator.classes)
        max_val = float(max(counter.values()))
        self.class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
        return self.class_weights

    def make_model(self):
        self.model = Sequential()

        self.model.add(Convolution2D(self.nb_filters,
                                self.kernel_size,input_shape=(self.img_rows,self.img_cols,self.input_dim)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Dropout(0.1))

        self.model.add(Convolution2D(self.nb_filters,
                                self.kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Dropout(0.1))

        self.model.add(Convolution2D(self.nb_filters,
                                self.kernel_size))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))
        self.model.add(Dropout(0.1))

        # transition to an mlp
        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(self.nb_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=0.00005),
                      metrics=['accuracy'])

    def fit(self):
        filepath='models/Simple_CNN.h5'
        mc = callbacks.ModelCheckpoint(filepath,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
        hist = callbacks.History()
        es = callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=4,
                                            verbose=1,
                                            mode='auto')
        if not os.path.exists('tensorboard_logs/Simple_CNN_tensorboard'):
            os.makedirs('tensorboard_logs/Simple_CNN_tensorboard')
        tensorboard = callbacks.TensorBoard(
                    log_dir='tensorboard_logs/Simple_CNN_tensorboard',
                    histogram_freq=0,
                    batch_size=self.batch_size,
                    write_graph=True,
                    embeddings_freq=0,
                    write_images=False)

        self._find_class_weights()

        history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=self.nb_epoch,
            class_weight=self.class_weights,
            validation_data=self.validation_generator,
            validation_steps=len(self.validation_generator),
            callbacks=[mc,hist,es,tensorboard])

        return history

def open_saved_model(model_name,generator):
    model = load_model(model_name)
    predictions = np.argmax(model.predict_generator(generator, steps=len(generator), verbose=1),axis=1)
    evaluation = model.evaluate_generator(generator, steps=len(generator), verbose=1)
    print(evaluation,'\n',predictions)

if __name__ == '__main__':
    Banana_CNN = SimpleCNN()
    _, _, holdout = Banana_CNN.make_generators('../data/Banana_People_Not/4_Classes')
    Banana_CNN.make_model()
    history = Banana_CNN.fit()
    open_saved_model('models/Simple_CNN.h5',holdout)

    from keras.utils import plot_model
    plot_model(Banana_CNN.model, to_file='../graphics/Simple_CNN_model.png')

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('../graphics/Simple_CNN_acc_hist.png')
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('../graphics/Simple_CNN_loss_hist.png')
    plt.close()
