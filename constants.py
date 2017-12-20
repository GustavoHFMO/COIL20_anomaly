#!/usr/local/bin/python

''' Anomaly class '''
ANOMALY_CLASS = 5

''' MNIST image format '''
MNIST_IMG_ROWS = 28
MNIST_IMG_COLS = 28
# The MNIST images are grayscale
MNIST_IMG_CHANNELS = 1

''' MNIST image format '''
CIFAR_IMG_ROWS = 32
CIFAR_IMG_COLS = 32
# The CIFAR10 images are RGB.
CIFAR_IMG_CHANNELS = 3

''' Inliers are labeled 1, while outliers are labeled 0. '''
ANOMALY_DATA_REPRESENTATION = 0
NORMAL_DATA_REPRESENTATION = 1

# ''' Inliers are labeled 1, while outliers are labeled -1. '''
# OCSVM_ANOMALY_DATA_REPRESENTATION = -1
# OCSVM_NORMAL_DATA_REPRESENTATION = 1

''' random state for replication'''
RANDOM_STATE = 1

''' CNN values'''
# number of convolutional filters to use
NB_FILTERS = 64
# convolution kernel size
NB_CONV = 5
# size of pooling area for max pooling
NB_POOL = 3
# stride of pooling operation
STRIDE_POOL = 2
# upsampling_size = stride_pool -> compression factor = decompression factor
UPSAMPLING_SIZE = STRIDE_POOL  
# number of epoch
NB_EPOCH = 100 #100
NB_EPOCH_MULTI_CLASS = 1 #20
# minibatch size
BATCH_SIZE=128
# CAE_deep
DEEP = 10

#optimizer function used on the training of the model
OPT_FUNC = 'adam'
# OPT_FUNC = 'sgd'


