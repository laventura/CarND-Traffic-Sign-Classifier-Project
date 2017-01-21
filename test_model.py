import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import time as time
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import random

import os
from PIL import Image
import time

import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.metrics import top_k_categorical_accuracy

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.regularizers import l2, activity_l2
from keras.models import model_from_yaml
from keras.models import model_from_json

# other Keras imports
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

INPUT_SHAPE = (32, 32, 3)
n_classes = 43
N_CLASSES = n_classes

print('TensorFlow version: ', tf.__version__)

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data
DATA_DIR = 'traffic-signs-data'

training_file = DATA_DIR + '/train.p'
testing_file =  DATA_DIR + '/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test,  y_test  = test['features'],  test['labels']

print('Train set shape:{0}'.format(X_train.shape))
print('Test  set shape:{0}'.format(X_test.shape))

print('Train labels: {0}'.format(y_train.shape))
print('Test  labels: {0}'.format(y_test.shape))

### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = np.shape(X_train[5])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

y_labels_train = np_utils.to_categorical(y_train, n_classes)
y_labels_test  = np_utils.to_categorical(y_test,  n_classes)
print('train labels: ', y_labels_train.shape)
print('test  labels: ', y_labels_test.shape)

signs = pd.read_csv('signnames.csv')
print('Total signs: ', len(signs))

# Equalization functions
def eq_histogram_rescale(image):
    ''' Equalize histogram (for each channel), then rescale to 1/255
    '''
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    image = image / 255.
    return image

def mean_subtract(img):
    ''' Remove mean (feature-wise)
    '''
    #img  = img / 255.
    img -= np.mean(img, axis=0)
    return img

def equalize_and_mean_subtract(image):
    ''' Equalize histograms, and then remove mean.
    '''
    # 1st equalise histo
    img_eh = eq_histogram_rescale(image)
    # then subtract mean
    return mean_subtract(img_eh)

# Pre-process the train / test data -- equalize Histogram
# Use this for Model Fitting..
X_train_proc = np.array([eq_histogram_rescale(X_train[i]) for i in range(len(X_train))], dtype=np.float32)
X_test_proc  = np.array([eq_histogram_rescale(X_test[i]) for i in range(len(X_test))], dtype=np.float32)

## Split Train-Validation 

from sklearn.model_selection import train_test_split

np.random.seed(123)

X_train, X_validation, y_train, y_validation = train_test_split(X_train_proc,
                                                               y_train,
                                                               random_state=42,
                                                               test_size=0.2)
print('Training shape:{0}'.format(X_train.shape))
print('Validatn shape:{0}'.format(X_validation.shape))
print('Testing  shape:{0}'.format(X_test_proc.shape))

print('Train   labels:{0}'.format(y_train.shape))
print('Valid.  labels:{0}'.format(y_validation.shape))
print('Test    labels:{0}'.format(y_test.shape))

## One-hot encode Labels
y_train_ohe      = np_utils.to_categorical(y_train, n_classes)
y_validation_ohe = np_utils.to_categorical(y_validation, n_classes)
y_test_ohe       = np_utils.to_categorical(y_test,  n_classes)
print('train labels: ', y_train_ohe.shape)
print('valid labels: ', y_validation_ohe.shape)
print('test  labels: ', y_test_ohe.shape)




def create_base_model():
    print('Creating baseline model...')
    model = Sequential()
    model.add(Convolution2D(30, 5, 5, input_shape=INPUT_SHAPE, activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(15, 3, 3, activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu' ))
    model.add(Dense(50, activation='relu' ))
    
    # final layer
    model.add(Dense(n_classes, activation='softmax' ))

    print(model.summary())
    return model

def create_lenet_model():
    print('Creating LeNet model...')
    model = Sequential()
    
    # L1
    model.add(Convolution2D(16, 5, 5, input_shape=INPUT_SHAPE, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))
    
    # L2
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    # FC layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))

    # final layer
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    return model

def create_lenet_model_fatFC():
    print('Creating LeNet-fatFC model...')
    model = Sequential()
    
    # L1
    model.add(Convolution2D(32, 5, 5, input_shape=INPUT_SHAPE, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.5))
    
    # L2
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    # FC layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))

    # final layer
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    return model

def create_vgg_lite():
    print('Creating VGG-lite1 model...')
    model = Sequential()
    
    # L1
    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='valid', input_shape=INPUT_SHAPE))
    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))
    
    # L2
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='valid'))
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    # FC layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))

    # final layer
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    return model


def create_lenet_model_3Conv():
    print('Creating LeNet_3Conv model...')
    model = Sequential()
    
    # L0
    model.add(Convolution2D(8, 5, 5, input_shape=INPUT_SHAPE, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))
    
    # L1
    model.add(Convolution2D(16, 5, 5, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    # L2
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    # FC layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))

    # final layer
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    return model

def create_lenet_model_4FC():
    print('Creating LeNet-4FC model...')
    model = Sequential()
    
    # L1
    model.add(Convolution2D(16, 5, 5, input_shape=INPUT_SHAPE, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))
    
    # L2
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    # FC layers
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))

    # final layer
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    return model



def create_lenet_model_L2():
    print('Creating LeNet model...')
    model = Sequential()
    
    # L1
    model.add(Convolution2D(16, 5, 5, input_shape=INPUT_SHAPE, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))
    
    # L2
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    # FC layers
    model.add(Dense(1024, activation='relu',  W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',  W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu',  W_regularizer=l2(0.01)))

    # final layer
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    return model



def create_deep_lenet_model():
    print('Creating Deep LeNet model...')
    model = Sequential()
    
    model.add(Convolution2D(4, 5, 5, input_shape=INPUT_SHAPE, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Convolution2D(8, 5, 5, input_shape=INPUT_SHAPE, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    
    model.add(Convolution2D(16, 3, 3, activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))

    # final layer
    model.add(Dense(n_classes, activation='softmax'))

    print(model.summary())
    return model

def deep_model1():
    print('Creating deep1 model...')
    model = Sequential()
    # L0
    model.add(Convolution2D(3, 1, 1, input_shape=INPUT_SHAPE, 
        activation='relu', 
        border_mode='same', 
        init='lecun_uniform'))

    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(32, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.3))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(64, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.3))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(128, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax' ))

    print(model.summary())
    return model

def deep_model3():
    print('Creating DEEP-3 model...')
    model = Sequential()
    # L0
    model.add(Convolution2D(3, 1, 1, input_shape=INPUT_SHAPE, 
        activation='relu', 
        border_mode='same', 
        init='lecun_uniform'))

    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='same', W_regularizer=l2(0.01)))
    model.add(Convolution2D(32, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='same', W_regularizer=l2(0.01)))
    model.add(Convolution2D(64, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.01)))
    model.add(Convolution2D(128, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu', W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax' ))

    print(model.summary())
    return model

def deep_model4():
    print('Creating DEEP-4 model...')
    model = Sequential()
    # L0
    model.add(Convolution2D(3, 1, 1, input_shape=INPUT_SHAPE, 
        activation='relu', 
        border_mode='same', 
        init='lecun_uniform'))

    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.01)))
    model.add(Convolution2D(32, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.01)))
    model.add(Convolution2D(64, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.01)))
    model.add(Convolution2D(128, 3, 3,  activation='relu', border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(2,2), border_mode='valid'))

    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', W_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax' ))

    print(model.summary())
    return model

def plot_history(history, name='model'):
     # plot history
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(name + ' model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()



###################################
def run_model(model, name='model', weights=None, nb_epochs=10):
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print()

    from keras.callbacks import ReduceLROnPlateau

    now = time.localtime()

    learning_rate = 0.01
    decay_rate = learning_rate / nb_epochs

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    if weights is not None:
        ''' load saved weights'''
        print('Loading saved weights: ', weights)
        model.load_weights(weights)

    # compile model
    model.compile(loss='categorical_crossentropy', 
                        optimizer=sgd,
                        metrics=['accuracy', 'top_k_categorical_accuracy'])

    # Training params
    BATCH_SIZE = 64
    NB_EPOCHS  = nb_epochs

    # Callback
    filepath= name + '_' + str(now.tm_hour) + '_' + str(now.tm_min) + "_weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor='val_acc', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode= max )
    # reduce LR on schedule
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

    # all callbacks
    callbacks_list = [checkpoint, reduce_lr]

    print('Starting training...')
    # train/fit the model
    history = model.fit(X_train, 
                        y_train,
                        validation_data=(X_validation, y_validation),
                        callbacks=callbacks_list,
                        batch_size=BATCH_SIZE,
                        nb_epoch=NB_EPOCHS,
                        verbose=2)

    print('Doing eval on TEST...')
    score = model.evaluate(X_test_proc, y_test_ohe, verbose=1)
    print()
    print('Metrics are: ', model.metrics_names)
    print('Test scores: ', score)

    plot_history(history, name )


def run_model_augmentation(model, name='model', weights=None, nb_epochs=10):
    ''' Run model, while augmenting training / validation data '''

    print('** Augmenting Training data...')
    # 1. Augment training data
    train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            fill_mode='nearest')

    train_datagen.fit(X_train)

    # 2. Augment validation data
    print('** Augmenting Validation data...')
    validation_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            fill_mode='nearest')

    validation_datagen.fit(X_validation)

    # 3. Prepare for training - Load saved weights, if needed
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print()

    from keras.callbacks import ReduceLROnPlateau

    now = time.localtime()

    learning_rate = 0.01
    decay_rate = learning_rate / nb_epochs

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    if weights is not None:
        ''' load saved weights'''
        print('Loading saved weights: ', weights)
        model.load_weights(weights)

    # compile model
    model.compile(loss='categorical_crossentropy', 
                        optimizer=sgd,
                        metrics=['accuracy', 'top_k_categorical_accuracy'])

    # Training params
    BATCH_SIZE = 64
    NB_EPOCHS  = nb_epochs

    # Callback
    filepath= name + '_' + str(now.tm_hour) + '_' + str(now.tm_min) \
            + "_weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, 
                                 monitor='val_acc', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode= max )
    # reduce LR on schedule
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

    # all callbacks
    callbacks_list = [checkpoint, reduce_lr]

    # 4. Start Training/Fit
    history = model.fit_generator(
            train_datagen.flow(X_train, y_train_ohe, batch_size=BATCH_SIZE),
            samples_per_epoch=X_train.shape[0],
            nb_epoch=NB_EPOCHS,
            validation_data=validation_datagen.flow(X_validation, y_validation_ohe,
                batch_size=BATCH_SIZE),
            nb_val_samples=len(X_validation),
            callbacks=callbacks_list,
            verbose=2)

    # 5. Test on test data
    print('Doing eval on TEST...')
    score = model.evaluate(X_test_proc, y_test_ohe, verbose=1)
    print()
    print('Metrics are: ', model.metrics_names)
    print('Test scores: ', score)

    plot_history(history, name)

def save_model(model, name='model', weights=None):
    ''' Save the model ''' 
    print('Saving model: ', name)

    model_yaml = model.to_yaml()
    with open(name + '.yaml', 'w') as yaml_file:
        yaml_file.write(model_yaml)

    if weights is not None:
        ''' load saved weights'''
        print('Loading saved weights: ', weights)
        model.load_weights(weights)
        print('model weights loaded')

    # compile model
    print('compiling model...')
    model.compile(loss='categorical_crossentropy', 
                        optimizer='adam',
                        metrics=['accuracy', 'top_k_categorical_accuracy'])


    #print('saving weights: ', name + '.hdf5')
    #model.save_weights(name + '.hdf5')


def list_top5(proba, y_true_classid):
    ''' proba is probability vector for one prediction; size: 43'''
    # get index of top5 probabilities, with max at the front
    top5_indices = np.argsort(proba)[::-1][:5]
    # find top5 classes from Sign Names
    top5_classes = signs.iloc[top5_indices]

    # correct sign
    true_sign = signs.iloc[y_true_classid]
    print('-- True label:\n  classId: {0} ===> {1}'
            .format(true_sign['ClassId'], true_sign['SignName']))
    print('-- top 5 proba --')
    # [print("%0.5f" % k) for k in proba[top5_indices]]
    # print(top5_classes)
    top5_classes['prob'] = proba[top5_indices]
    print(top5_classes)
    
    '''
    for r in range(len(top5_indices)):
        row = top5_classes[r]
        id  = row['ClassId']
        nam = row['SignName']
        p   = proba[top5_indices][r]
        print("  %0.5f --> %d --> %s" % (p, id, nam))
    '''



def do_predictions(model, X):
    ''' do some sample predictions on X'''

    print('Doing predictions..')
    print('Model metrics:', model.metrics_names)

    #scores = model.evaluate(X_test_proc, y_test_ohe, verbose=1)
    #print('Test Scores:', scores)

    len5 = len(X) - 10
    # get a random index
    randix = np.random.randint(0, high=len5)

    randix = random.randint(0, len5)

    print('randix: ', randix)

    print('Doing predictions on 5 samples')
    # get a sample of 5 images, starting at randix
    x_small = X[randix: randix+5]

    # predictions = model.predict(x_small, batch_size=5, verbose=1)
    # print(predictions)

    print('Doing proba predictions on 5 samples')
    proba = model.predict_proba(x_small, batch_size=5, verbose=1)
    print(proba.shape)
    # print(proba)

    print('Doing class predictions on 5 samples')
    pred_classes = model.predict_classes(x_small, batch_size=5, verbose=1)
    print(pred_classes.shape)
    print('predicted classes: ', pred_classes)

    for i in range(len(proba)):
        y_true_classid = y_test[randix + i]   # true y
        list_top5(proba[i], y_true_classid)




####### Baseline model
#baseline_model = create_base_model()
#run_model(baseline_model, name='baseline')

## leNet -- 95-96% acc w/ 25 epochs
#lenet_model = create_lenet_model()
#saved_weights = 'lenet_12_11_weights-17-0.99.hdf5'
#run_model(lenet_model, name='lenet', weights=None, nb_epochs=20)

## leNet - fatFC
#lenet_model_fatFC = create_lenet_model_fatFC()
#saved_weights = 'lenet_12_11_weights-17-0.99.hdf5'
#run_model(lenet_model_fatFC, name='lenet_fatFC', weights=None, nb_epochs=20)

## VGG lite -- 98% test accuracy after 35 epochs
vgg_lite2 = create_vgg_lite()
#saved_weights = 'vgg_lite2_11_56_weights-14-0.97.hdf5'  # had 98% accuracy after 35 epochs
saved_weights = 'vgg_lite2_14_39_weights-17-0.98.hdf5'
#run_model(vgg_lite1, name='vgg_lite1', weights=saved_weights, nb_epochs=15)
#run_model_augmentation(vgg_lite2,  name='vgg_lite2', weights=None, nb_epochs=20) 
save_model(vgg_lite2, name='vgg_lite2_14_39', weights=saved_weights)

do_predictions(vgg_lite2, X_test_proc)

## leNet 3Conv
#lenet_model_3conv = create_lenet_model_3Conv()
#saved_weights = 'lenet_3conv_15_3_weights-19-0.95.hdf5'
#run_model(lenet_model_3conv, name='lenet_3conv', weights=None, nb_epochs=20)

# leNet - 4FC -- was 96% acc w/ 35 epochs
#lenet_model_4FC = create_lenet_model_4FC()
#saved_weights = 'lenet-4FC_14_25_weights-08-0.99.hdf5'
#run_model(lenet_model_4FC, name='lenet-4FC', weights=saved_weights, nb_epochs=10)

## leNet with L2
#lenet_L2 = create_lenet_model_L2()
#saved_weights = 'lenet-L2_13_45_weights-14-0.96.hdf5'
#run_model(lenet_L2, name='lenet-L2', weights=saved_weights, nb_epochs=15)

#deep_lenet = create_deep_lenet_model()
#run_model(deep_lenet, name='DEEP-LeNet')

print()
### Deep Model 
#deep1 = deep_model1()
#run_model(deep1, name='deep1')

### Deep Model 
#deep3 = deep_model3()
#run_model(deep3, name='deep3')

### Deep Model 
#deep4 = deep_model4()
#run_model(deep4, name='deep4')
