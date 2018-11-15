import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import numpy as np
from PIL import Image
from keras.applications import Xception, ResNet50
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.models import Model
from keras.applications.xception import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras import callbacks
from print_pretty_confusion_matrix import plot_confusion_matrix_from_data
import matplotlib.pyplot as plt
mpl.rcParams.update({
    'figure.figsize'      : (10,50),
    # 'font.size'           : 20.0,
    # 'axes.titlesize'      : 'large',
    # 'axes.labelsize'      : 'medium',
    # 'xtick.labelsize'     : 'medium',
    # 'ytick.labelsize'     : 'medium',
    # 'legend.fontsize'     : 'large',
    # 'legend.loc'          : 'upper right'
})


class TranseferModel():

    def __init__(self,model=Xception,target_size=(299,299),weights='imagenet',
                n_categories=4,batch_size=8,augmentation_strength=0.2,
                preprocessing=preprocess_input,epochs=10):
        self.model = model
        self.target_size = target_size
        self.input_size = self.target_size + (3,)
        self.weights = weights
        self.n_categories = n_categories
        self.batch_size = batch_size
        self.train_generator = None
        self.validation_generator = None
        self.holdout_generator = None
        self.augmentation_strength = augmentation_strength
        self.preprocessing = preprocessing
        self.epochs = epochs

    def _create_transfer_model(self):
        base_model = self.model(weights=self.weights,
                          include_top=False,
                          input_shape=self.input_size)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.n_categories, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        return self.model

    def make_generators(self,directory):
        train_data_dir = directory+'/train'
        validation_data_dir = directory+'/validation'
        holdout_data_dir = directory+'/holdout'

        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing,
            rotation_range=15*self.augmentation_strength,
            width_shift_range=self.augmentation_strength,
            height_shift_range=self.augmentation_strength,
            shear_range=self.augmentation_strength,
            zoom_range=self.augmentation_strength,
            horizontal_flip=True,
            vertical_flip=True)

        test_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocessing)

        self.train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical')

        self.validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical')

        self.holdout_generator = test_datagen.flow_from_directory(
            holdout_data_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False)
        return self.train_generator, self.validation_generator, self.holdout_generator

    def fit(self,freeze_indices,optimizers,warmup_epochs=5):
        #change head
        self._create_transfer_model()
        self.change_trainable_layers(freeze_indices[0])
        # train head
        self.model.compile(optimizer=optimizers[0],
                      loss='categorical_crossentropy', metrics=['accuracy'])
        # callbacks
        mc = callbacks.ModelCheckpoint('transfer_CNN.h5',
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
        hist = callbacks.History()
        es = callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0,
                                            patience=1,
                                            verbose=0,
                                            mode='auto')
        if not os.path.exists('transfer_CNN_tensorboard'):
            os.makedirs('transfer_CNN_tensorboard')
        tensorboard = callbacks.TensorBoard(
                    log_dir='transfer_CNN_tensorboard',
                    histogram_freq=0,
                    batch_size=self.batch_size,
                    write_graph=True,
                    embeddings_freq=0)

        history = self.model.fit_generator(self.train_generator,
                                      steps_per_epoch=len(self.train_generator),
                                      epochs=warmup_epochs,
                                      validation_data=self.validation_generator,
                                      validation_steps=len(self.validation_generator),
                                      callbacks=[mc, tensorboard, es, hist])
        # train more layers
        self.change_trainable_layers(freeze_indices[1])

        self.model.compile(optimizer=optimizers[0],
                      loss='categorical_crossentropy', metrics=['accuracy'])

        history = self.model.fit_generator(self.train_generator,
                                      steps_per_epoch=len(self.train_generator),
                                      epochs=self.epochs,
                                      validation_data=self.validation_generator,
                                      validation_steps=len(self.validation_generator),
                                      callbacks=[mc, tensorboard, es, hist])

    def change_trainable_layers(self, trainable_index):
        for layer in self.model.layers[:trainable_index]:
            layer.trainable = False
        for layer in self.model.layers[trainable_index:]:
            layer.trainable = True

    def best_training_model(self):
        model = load_model('transfer_CNN.h5')
        predictions = model.predict_generator(self.holdout_generator,
                                                steps=len(self.holdout_generator))
        predictions = np.argmax(predictions, axis=-1)
        label_map = (self.holdout_generator.class_indices)
        label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
        names = np.array([x.split('/') for x in self.holdout_generator.filenames])
        predictions = [label_map[k] for k in predictions]
        predictions = np.array(predictions).reshape(-1,1)
        data = np.hstack((names,predictions))
        metrics = model.evaluate_generator(self.holdout_generator,
                                            steps=len(self.holdout_generator))
        return metrics, data

    def print_matrix(self,y_true,y_pred):
        columns = self.holdout_generator.class_indices.keys()
        plot_confusion_matrix_from_data(y_true,y_pred,columns=columns)
        return

    def return_failed_images(self,dir,data):
        failed = data[data[:,0]!=data[:,2]]
        fig, axes = plt.subplots(len(failed),2)
        for idx,row in enumerate(failed):
            file_path = dir+'/holdout/'+row[0]+'/'+row[1]
            # plot original image
            axes[idx][0].imshow(Image.open(file_path))
            # recreate preprocessed image
            img = image.load_img(file_path, target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            axes[idx][1].imshow(img)
            # set title and remove ticks
            axes[idx].set_title(row[2])
            axes[idx].get_xaxis().set_visible(False)
            axes[idx].get_yaxis().set_visible(False)
            for t in axes[idx].xaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False
            for t in axes[idx].yaxis.get_major_ticks():
                t.tick1On = False
                t.tick2On = False
            # for idx2 in range(2):
            #     axes[idx][idx2].set_title(row[2])
            #     axes[idx][idx2].get_xaxis().set_visible(False)
            #     axes[idx][idx2].get_yaxis().set_visible(False)
            #     for t in axes[idx][idx2].xaxis.get_major_ticks():
            #         t.tick1On = False
            #         t.tick2On = False
            #     for t in axes[idx][idx2].yaxis.get_major_ticks():
            #         t.tick1On = False
            #         t.tick2On = False
        plt.tight_layout()
        plt.savefig('failed_images.png')
        plt.close()
        return

if __name__ == '__main__':
    dir = '../data/Banana_People_Not/4_Classes'
    transfer_CNN = TranseferModel()
    transfer_CNN.make_generators(dir)

    freeze_indices = [132, 126]
    optimizers = [Adam(lr=0.0006), Adam(lr=0.0001)]

    # transfer_CNN.fit(freeze_indices,optimizers)
    metrics, data = transfer_CNN.best_training_model()
    # transfer_CNN.print_matrix(data[:,0],data[:,2])
    print(metrics)
    transfer_CNN.return_failed_images(dir,data)
