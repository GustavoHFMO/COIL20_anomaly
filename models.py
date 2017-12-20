#!/usr/local/bin/python

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Flatten,Activation, Deconv2D, Reshape, Cropping2D, ZeroPadding2D
from keras.models import Model, Sequential
from keras.datasets import mnist, cifar10
import numpy as np

from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.optimizers import SGD

from keras import initializers

from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV

import util
import constants
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras import regularizers
from keras.legacy.layers import merge
from keras.layers.merge import Concatenate

from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3


def convnet_autoencoder_deep1(X_train, dataset_name):
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_convnet_autoencoder_deep1')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    encoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     rms = RMSprop()
#     autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    
#     autoencoder.fit(X_train, X_train,
#                     nb_epoch=30,
#                     batch_size=128,
#                     shuffle=True,
#                     validation_data=(X_test, X_test))
    return autoencoder, encoded, decoded, input_img

def convnet_autoencoder_deep3(X_train, dataset_name):
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_convnet_autoencoder_deep3')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    ''' 5 is the max_number of upsamplings '''
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100')):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='valid')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     rms = RMSprop()
#     autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, encoded, decoded, input_img

def convnet_autoencoder_deep5(X_train, dataset_name):
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
    elif((dataset_name == 'COIL20') or (dataset_name == 'CIFAR_gray')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.MNIST_IMG_CHANNELS))
        #input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_convnet_autoencoder_deep5')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    ''' 5 is the max_number of upsamplings '''
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    elif((dataset_name == 'COIL20') or (dataset_name == 'CIFAR_gray')):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
        
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='valid')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
        
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     rms = RMSprop()
#     autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, encoded, decoded, input_img

def convnet_autoencoder_deep10(X_train, dataset_name):
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
    elif((dataset_name == 'COIL20') or (dataset_name == 'CIFAR_gray')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.MNIST_IMG_CHANNELS))
        #input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_convnet_autoencoder_deep10')
    
#     #test1
#     init = initializers.RandomNormal(mean=0.0, stddev=0.01)
#     init_bias = initializers.RandomNormal(mean=0.0, stddev=0.1)
#     #test2
#     init = initializers.RandomNormal(mean=0.0, stddev=0.001)
#     init_bias = initializers.RandomNormal(mean=0.0, stddev=0.01)
# #     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', kernel_initializer=init, bias_initializer=init_bias)(x)
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    ''' 5 is the max_number of upsamplings '''
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)#1
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#2
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#3
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#4
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#5
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#6
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#7
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#8
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#9
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#10
        decoded = Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
        
    elif((dataset_name == 'COIL20') or (dataset_name == 'CIFAR_gray')):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)#1
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#2
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#3
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#4
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#5
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#6
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#7
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#8
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#9
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#10
        decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
        
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)#1
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#2
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#3
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#4
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#5
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#6
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)#7
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#8
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='valid')(x)#9
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)#10
        decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)

#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
#     x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='valid')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     decoded = Conv2D(1, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
#     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#     sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True)
    
#     autoencoder.compile(loss='binary_crossentropy', optimizer=sgd)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     rms = RMSprop()
#     autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, encoded, decoded, input_img

def convnet_autoencoder_deep1_split_channel(X_train, dataset_name):
    input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, 1))
    print(dataset_name + '_convnet_autoencoder_deep1_split_channel')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    encoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    decoded = Conv2D(1, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    

    return autoencoder, encoded, decoded, input_img

def convnet_autoencoder_deep10_split_channel(X_train, dataset_name):
    input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, 1))
    print(dataset_name + '_convnet_autoencoder_deep10_split_channel')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    ''' 5 is the max_number of upsamplings '''
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    decoded = Conv2D(1, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     rms = RMSprop()
#     autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, encoded, decoded, input_img

def convnet_autoencoder_deep1_bn(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print(dataset_name + '_convnet_autoencoder_bn_deep1')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    return model

def convnet_autoencoder_deep1_dropout(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print(dataset_name + '_convnet_autoencoder_dropout_deep1')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    return model

def convnet_autoencoder_deep1_bn_augmentation(X_train, dataset_name):
    data_augmentation = True
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print(dataset_name + '_convnet_autoencoder_bn_deep1_augmentation')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
#     model.fit(X_train, X_train,
#                     epochs=constants.NB_EPOCH,
#                     batch_size=constants.BATCH_SIZE)
    
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, X_train,
                  epochs=constants.NB_EPOCH,
                  batch_size=constants.BATCH_SIZE,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        
        print('Fitting CNN.')
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, X_train,
                                         batch_size=constants.BATCH_SIZE),
                            steps_per_epoch=X_train.shape[0] // constants.BATCH_SIZE,
                            epochs=constants.NB_EPOCH)
    
    return model

def convnet_autoencoder_deep1_augmentation(X_train, dataset_name):
    data_augmentation = True
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print(dataset_name + '_convnet_autoencoder_bn_deep1_augmentation')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
#     model.fit(X_train, X_train,
#                     epochs=constants.NB_EPOCH,
#                     batch_size=constants.BATCH_SIZE)
    
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, X_train,
                  epochs=constants.NB_EPOCH,
                  batch_size=constants.BATCH_SIZE,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        
        print('Fitting CNN.')
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, X_train,
                                         batch_size=constants.BATCH_SIZE),
                            steps_per_epoch=X_train.shape[0] // constants.BATCH_SIZE,
                            epochs=constants.NB_EPOCH)
    
    return model


def convnet_autoencoder_deep10_bn(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print(dataset_name + '_convnet_autoencoder_bn_deep10')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' nao pode mais BN''' 
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        model.add(Conv2D(constants.NB_FILTERS, (3, 3), padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    return model

def convnet_autoencoder_deep10_dropout(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print(dataset_name + '_convnet_autoencoder_dropout_deep10')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
    
    ''' nao pode mais BN''' 
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    model.add(Dropout(0.25))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        model.add(Conv2D(constants.NB_FILTERS, (3, 3), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
        model.add(Dropout(0.25))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    return model

def convnet_autoencoder_deep10_bn_augmentation(X_train, dataset_name):
    data_augmentation = True
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print(dataset_name + '_convnet_autoencoder_bn_deep10_augmentation')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' nao pode mais BN''' 
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        model.add(Conv2D(constants.NB_FILTERS, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, X_train,
                  epochs=constants.NB_EPOCH,
                  batch_size=constants.BATCH_SIZE,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        
        print('Fitting CNN.')
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, X_train,
                                         batch_size=constants.BATCH_SIZE),
                            steps_per_epoch=X_train.shape[0] // constants.BATCH_SIZE,
                            epochs=constants.NB_EPOCH)
    return model

def convnet_autoencoder_deep10_augmentation(X_train, dataset_name):
    data_augmentation = True
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print(dataset_name + '_convnet_autoencoder_deep10_augmentation')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' nao pode mais BN''' 
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#     model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        model.add(Conv2D(constants.NB_FILTERS, (3, 3), padding='same'))
#         model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, X_train,
                  epochs=constants.NB_EPOCH,
                  batch_size=constants.BATCH_SIZE,
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        
        print('Fitting CNN.')
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, X_train,
                                         batch_size=constants.BATCH_SIZE),
                            steps_per_epoch=X_train.shape[0] // constants.BATCH_SIZE,
                            epochs=constants.NB_EPOCH)
    return model

def cnn_deep1_sequential(X_train, y_train):
#     input_img = Input(shape=(28, 28, 1))
    input_img = (28, 28, 1)
    print('cnn_deep1_sequential')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same', input_shape=input_img))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE)
    return model

def cnn_deep1_model(X_train, y_train):
    input_img = Input(shape=(28, 28, 1))
#     input_img = (28, 28, 1)
    print('cnn_deep1_model')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
               activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs= input_img, outputs=out)
    
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE)
    return model

def cnn_deep1_model_and_features(X_train, y_train):
    input_img = Input(shape=(28, 28, 1))
#     input_img = (28, 28, 1)
    print('cnn_deep1_model')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
               activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL))(x)
    x = Flatten()(x)
    feat = Dense(128, activation='relu')(x)
#     x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(feat)
    
    model = Model(inputs= input_img, outputs=out)
    
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=1,
                    batch_size=constants.BATCH_SIZE)
    return model, feat

def cnn_binary_clf_validation(X_train, y_train, x_valid, y_valid, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print('cnn_sequential_' + dataset_name)
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same', input_shape=input_img))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Flatten())
    model.add(Dense(700, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    validation_data=(x_valid, y_valid))
    return model

def cnn_binary_clf(X_train, y_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print('cnn_sequential_' + dataset_name)
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same', input_shape=input_img))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Flatten())
    model.add(Dense(700, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss=losses.binary_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE)
    return model

def multiclass_cnn(X_train, y_train, dataset_name, num_classes):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print('cnn_multi_clf_' + dataset_name)
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same', input_shape=input_img))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Flatten())
    model.add(Dense(700, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=constants.NB_EPOCH_MULTI_CLASS,
                    batch_size=constants.BATCH_SIZE)
    return model

def multiclass_cnn_dropout(X_train, y_train, dataset_name, num_classes):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print('multiclass_cnn_dropout_' + dataset_name)
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', input_shape=input_img))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(700, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=constants.NB_EPOCH_MULTI_CLASS,
                    batch_size=constants.BATCH_SIZE)
    return model

def multiclass_cnn_bn(X_train, y_train, dataset_name, num_classes):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print('multiclass_cnn_bn_' + dataset_name)
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    
    model.add(Flatten())
    model.add(Dense(700))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=constants.NB_EPOCH_MULTI_CLASS,
                    batch_size=constants.BATCH_SIZE)
    return model

def multiclass_cnn_bn_dropout(X_train, y_train, dataset_name, num_classes):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    print('multiclass_cnn_bn_dropout_' + dataset_name)
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(700))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train,
                    epochs=constants.NB_EPOCH_MULTI_CLASS,
                    batch_size=constants.BATCH_SIZE)
    return model


def convnet_autoencoder_deep1_bn_invertido(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_CHANNELS, constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_CHANNELS, constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS)
    print(dataset_name + '_convnet_autoencoder_bn_deep1_invertido')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', data_format='channels_first', input_shape=input_img))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', data_format='channels_first', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', data_format='channels_first', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    return model

def convnet_autoencoder_deep10_bn_invertido(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_CHANNELS, constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_CHANNELS, constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS)
    print(dataset_name + '_convnet_autoencoder_bn_deep10_invertido')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', data_format='channels_first', input_shape=input_img))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
    
    ''' nao pode mais BN''' 
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), data_format='channels_first', padding='same'))
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', data_format='channels_first', padding='same'))
     
    elif(dataset_name == 'MNIST'):
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        model.add(Conv2D(constants.NB_FILTERS, (3, 3), data_format='channels_first', padding='same'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE), data_format='channels_first'))
         
        model.add(Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), data_format='channels_first', activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    return model

def convnet_autoencoder_deep10_init(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_init_convnet_autoencoder_deep10')
    
    #test1
    init = initializers.RandomNormal(mean=0.0, stddev=0.01)
    init_bias = initializers.RandomNormal(mean=0.0, stddev=0.1)
    
#     #test2
#     init = initializers.RandomNormal(mean=0.0, stddev=0.001)
#     init_bias = initializers.RandomNormal(mean=0.0, stddev=0.01)
# #     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
    encoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='valid', use_bias=True, kernel_initializer=init, bias_initializer=init_bias)(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, encoded, decoded, input_img

def convdeconv_autoencoder_deep10(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_deconvnet_autoencoder_deep10')
    
#     #test1
#     init = initializers.RandomNormal(mean=0.0, stddev=0.01)
#     init_bias = initializers.RandomNormal(mean=0.0, stddev=0.1)
#     #test2
#     init = initializers.RandomNormal(mean=0.0, stddev=0.001)
#     init_bias = initializers.RandomNormal(mean=0.0, stddev=0.01)
# #     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same', kernel_initializer=init, bias_initializer=init_bias)(x)
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    ''' 5 is the max_number of upsamplings '''
    if(dataset_name == 'CIFAR'):
#         x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#         x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#         x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#         x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#         x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#         decoded = Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
        decoded = Deconv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
        x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='valid')(x)
        x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
        decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     # The next layer needs filter_size = 3x3 and border_mode='valid' to have shape (None, 14, 14, 64) 
#     x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='valid')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     decoded = Conv2D(1, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
#     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#     sgd = SGD(lr=0.1, decay=0.0005, momentum=0.9, nesterov=True)
    
#     autoencoder.compile(loss='binary_crossentropy', optimizer=sgd)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     rms = RMSprop()
#     autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, encoded, decoded, input_img

def cae_flatten(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_flatten')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dense(1024)(x)
#     x = Dense(4*4*64)(x)
    x = Reshape((4,4,64))(x)
     
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    decoded = Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     rms = RMSprop()
#     autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img

# def convdeconv_cae_flatten(X_train, dataset_name):
#     if(dataset_name == 'CIFAR'):
#         input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
#     elif(dataset_name == 'MNIST'):
#         input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
# #     input_img = Input(shape=(28, 28, 1))
#     print(dataset_name + '_convdeconv_cae_flatten')
#     
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
#     x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
#     
#     x = Flatten()(x)
#     x = Dense(1024)(x)
#     x = Dense(512)(x)
#     x = Dense(1024)(x)
# #     x = Dense(4*4*64)(x)
#     x = Reshape((4,4,64))(x)
#      
#     x = Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     x = Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     x = Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
#     x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
#     decoded = Deconv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
#     
#     autoencoder = Model(input_img, decoded)
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     
#     autoencoder.fit(X_train, X_train,
#                     epochs=constants.NB_EPOCH,
#                     batch_size=constants.BATCH_SIZE,
#                     shuffle=True)
# 
#     return autoencoder, 0, decoded, input_img

def convdeconv_cae_flatten(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = (constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS)
    elif(dataset_name == 'MNIST'):
        input_img = (constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
        
#     if(dataset_name == 'CIFAR'):
#         input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
#     elif(dataset_name == 'MNIST'):
#         input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_sequential_convdeconv_cae_flatten')
    
    model = Sequential()
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same', input_shape=input_img))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
     
    model.add(Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL), padding='same'))
    
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Reshape((4,4,64)))
    
    model.add(Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
    
    model.add(Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
    
    model.add(Deconv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE)))
    
    model.add(Deconv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    model.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    return model, 0, 0, input_img
    
    
def cae_flatten_grayscale(X_train, dataset_name):
#     if(dataset_name == 'CIFAR'):
    input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     elif(dataset_name == 'MNIST'):
#         input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_flatten_grayscale')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dense(512)(x)
    x = Dense(1024)(x)
#     x = Dense(4*4*64)(x)
    x = Reshape((4,4,64))(x)
     
#     x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(encoded)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    decoded = Conv2D(constants.MNIST_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img


def cae_flatten_deep10(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_flatten_deep10')
    
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    decoded = MaxPooling2D(pool_size=(constants.NB_POOL, constants.NB_POOL), strides=(constants.STRIDE_POOL, constants.STRIDE_POOL),padding='same')(x)
    
    
    x = Flatten()(x)
    x = Dense(1*1*64)(x)
    x = Reshape((1,1,64))(x)
    
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    x = Conv2D(constants.NB_FILTERS, (constants.NB_CONV, constants.NB_CONV), activation='relu', padding='same')(x)
    x = UpSampling2D((constants.UPSAMPLING_SIZE, constants.UPSAMPLING_SIZE))(x)
    decoded = Conv2D(constants.CIFAR_IMG_CHANNELS, (constants.NB_CONV, constants.NB_CONV), activation='sigmoid', padding='same')(x)
#      
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
#     rms = RMSprop()
#     autoencoder.compile(optimizer=rms, loss='mean_squared_error')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img

def convnet_autoencoder_no_pool(X_train, dataset_name):
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
    elif((dataset_name == 'COIL20') or (dataset_name == 'CIFAR_gray')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.MNIST_IMG_CHANNELS))
    print(dataset_name + '_cae_no_pool')
    
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
#     x = Flatten()(x)
#     x = Dense(1024)(x)
#     x = Dense(512)(x)
#     x = Dense(1024)(x)
# #     x = Dense(4*4*64)(x)
#     x = Reshape((4,4,64))(x)
     
    x = Deconv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(192, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    x = Deconv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(96, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        decoded = Deconv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        decoded = Deconv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    elif((dataset_name == 'COIL20') or (dataset_name == 'CIFAR_gray')):
        decoded = Deconv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img


def convnet_autoencoder_no_pool_artigo(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_no_pool_artigo')

    ''' encoder'''
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    ''' bottleneck'''
    x = Conv2D(192, (1, 1), activation='relu', padding='same')(x)
    
    ''' decoder'''
    x = Deconv2D(192, (3, 3), activation='relu', padding='same')(x)
     
    x = Deconv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(192, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    x = Deconv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(96, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if(dataset_name == 'CIFAR'):
        decoded = Deconv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        decoded = Deconv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img

def convnet_autoencoder_no_pool_artigo_invertido(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_no_pool_artigo_invertido')

    ''' encoder'''
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(192, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(96, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    ''' bottleneck'''
    x = Conv2D(96, (1, 1), activation='relu', padding='same')(x)
    
    ''' decoder'''
    x = Deconv2D(96, (3, 3), activation='relu', padding='same')(x)
     
    x = Deconv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(96, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    x = Deconv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(192, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if(dataset_name == 'CIFAR'):
        decoded = Deconv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        decoded = Deconv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img

def convnet_autoencoder_no_pool_layers(X_train, dataset_name):
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
    elif((dataset_name == 'COIL20') or (dataset_name == 'CIFAR_gray')):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_no_pool_3l_64')
    
    ''' encoder '''
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    ''' decoder''' 
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        x = Cropping2D(cropping=((1, 1), (1, 1)))(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    #x = Cropping2D(cropping=((1, 1), (1, 1)))(x)

    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        decoded = Deconv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert') or (dataset_name == 'COIL20')):
        decoded = Deconv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     autoencoder.compile(optimizer='adam', loss='mse')
    
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img    

def convnet_autoencoder_no_pool_layers_valid(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_no_pool_3l_64_valid')
    
    ''' encoder '''
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu')(input_img)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2))(x)
    
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2))(x)

    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2))(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    ''' decoder''' 
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2))(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2))(x)
    
    if(dataset_name == 'MNIST'):
        x = Cropping2D(cropping=((1, 1), (1, 1)))(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2))(x)
        
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    #x = Cropping2D(cropping=((1, 1), (1, 1)))(x)

    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if(dataset_name == 'CIFAR'):
        decoded = Deconv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        decoded = Deconv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     autoencoder.compile(optimizer='adam', loss='mse')
    
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img 

def denoising_convnet_autoencoder_no_pool_layers(X_train, X_train_noisy, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_no_pool_3l_64_denoising')
    
    ''' encoder '''
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    ''' decoder''' 
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    #x = Cropping2D(cropping=((1, 1), (1, 1)))(x)

    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if(dataset_name == 'CIFAR'):
        decoded = Deconv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        decoded = Deconv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    
    autoencoder.fit(X_train_noisy, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img   

def convnet_autoencoder_no_pool_layers_sparse(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_no_pool_3l_64_sparse')
    
    ''' encoder '''
#     keras.layers.convolutional.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', 
#                                       data_format=None, dilation_rate=(1, 1), activation=None, 
#                                       use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
#                                       kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
#                                       kernel_constraint=None, bias_constraint=None)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(input_img)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    ''' decoder''' 
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same', bias_initializer='glorot_uniform',
                                    kernel_regularizer=regularizers.l1(10e-5), bias_regularizer=regularizers.l1(10e-5), activity_regularizer=regularizers.l1(10e-5))(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    ''' fim sparse: se 3l usar apenas 3 camadas '''
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    #x = Cropping2D(cropping=((1, 1), (1, 1)))(x)

    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if(dataset_name == 'CIFAR'):
        decoded = Deconv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        decoded = Deconv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    
    autoencoder.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)

    return autoencoder, 0, decoded, input_img 

def get_unet(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_unet')
    
#     inputs = Input((20, ISZ, ISZ))
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(input_img)
    conv1 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    
#     up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    ''' tem que ver se esse eixo eh 1 ou 3'''
    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    
#     up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
    
#     up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    
#     up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1]) 
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    
#     conv10 = Conv2D(N_Cls, 1, 1, activation='sigmoid')(conv9)
    conv10 = Conv2D(3, 1, 1, activation='sigmoid')(conv9)
    
    model = Model(input=input_img, output=conv10)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, X_train,
                    epochs=constants.NB_EPOCH,
                    batch_size=constants.BATCH_SIZE,
                    shuffle=True)
    
    return model, 0, 0, input_img


def convnet_autoencoder_no_pool_layers_zca(X_train, dataset_name):
    if(dataset_name == 'CIFAR'):
        input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
    elif(dataset_name == 'MNIST'):
        input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     input_img = Input(shape=(28, 28, 1))
    print(dataset_name + '_cae_no_pool_3l_64')
    
    ''' encoder '''
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Conv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    ''' decoder''' 
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)

    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
     
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    #x = Cropping2D(cropping=((1, 1), (1, 1)))(x)

    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', padding='same')(x)
    #x = Deconv2D(constants.NB_FILTERS, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
    
    if(dataset_name == 'CIFAR'):
        decoded = Deconv2D(3, (3, 3), activation='tanh', padding='same')(x)
    elif(dataset_name == 'MNIST'):
        decoded = Deconv2D(1, (3, 3), activation='tanh', padding='same')(x)
    
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     autoencoder.compile(optimizer='adam', loss='mse')
    
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True)
    
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)
        
    print('Fitting CNN.')
    # Fit the model on the batches generated by datagen.flow().
    autoencoder.fit_generator(datagen.flow(X_train, X_train,
                                     batch_size=constants.BATCH_SIZE),
                        steps_per_epoch=X_train.shape[0] // constants.BATCH_SIZE,
                        epochs=constants.NB_EPOCH)
    
#     autoencoder.fit(X_train, X_train,
#                     epochs=constants.NB_EPOCH,
#                     batch_size=constants.BATCH_SIZE,
#                     shuffle=True)

    return autoencoder, 0, decoded, input_img 

# def get_model(weights='imagenet'):
#     # create the base pre-trained model
#     base_model = InceptionV3(weights=weights, include_top=False)
# 
#     # add a global spatial average pooling layer
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     # let's add a fully-connected layer
#     x = Dense(1024, activation='relu')(x)
#     # and a logistic layer -- let's say we have 2 classes
#     predictions = Dense(len(data.classes), activation='softmax')(x)
# 
#     # this is the model we will train
#     model = Model(inputs=base_model.input, outputs=predictions)
#     return model

# def CAE_VGG(X_train, dataset_name):
#     if(dataset_name == 'CIFAR'):
#         input_img = Input(shape=(constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.CIFAR_IMG_CHANNELS))
#     elif(dataset_name == 'MNIST'):
#         input_img = Input(shape=(constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS))
#     print(dataset_name + '_CAE_VGG')
#     
#     ''' Decoder '''
#     x = ZeroPadding2D(((96, 96), (96, 96)))(input_img)
#     x = Conv2D(64, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(64, 3, 3, activation='relu')(x)
#     x = MaxPooling2D((2,2), strides=(2,2))(x)
#  
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(128, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(128, 3, 3, activation='relu')(x)
#     x = MaxPooling2D((2,2), strides=(2,2))(x)
#  
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(256, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(256, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(256, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(256, 3, 3, activation='relu')(x)
#     x = MaxPooling2D((2,2), strides=(2,2))(x)
#  
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(512, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(512, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(512, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(512, 3, 3, activation='relu')(x)
#     x = MaxPooling2D((2,2), strides=(2,2))(x)
#  
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(512, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(512, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(512, 3, 3, activation='relu')(x)
#     x = ZeroPadding2D((1,1))(x)
#     x = Conv2D(512, 3, 3, activation='relu')(x)
#     x = MaxPooling2D((2,2), strides=(2,2))(x)
# 
#     ''' Decoder '''
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(512, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(512, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(512, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(512, 3, 3, activation='relu')(x)
#     x = UpSampling2D((2,2))(x)
#     
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(512, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(512, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(512, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(512, 3, 3, activation='relu')(x)
#     x = UpSampling2D((2,2))(x)
#     
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(256, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(256, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(256, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(256, 3, 3, activation='relu')(x)
#     x = UpSampling2D((2,2))(x)
#     
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(128, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(128, 3, 3, activation='relu')(x)
#     x = UpSampling2D((2,2))(x)
#     
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(64, 3, 3, activation='relu')(x)
# #     x = ZeroPadding2D((1,1))(x)
#     x = Cropping2D(cropping=((1, 1), (1, 1)))(x)
#     x = Deconv2D(64, 3, 3, activation='relu')(x)
#     x = UpSampling2D((2,2))(x)
#     
#     x = Cropping2D(cropping=((81, 81), (81, 81)))(x)
#     
#     decoded = Deconv2D(3, (3, 3), activation='sigmoid')(x)
#     
#     
#     ''' Model '''
#     autoencoder = Model(input_img, decoded)
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#         
#     autoencoder.fit(X_train, X_train,
#                     epochs=constants.NB_EPOCH,
#                     batch_size=constants.BATCH_SIZE,
#                     shuffle=True)
# 
#     return autoencoder, 0, decoded, input_img
    
