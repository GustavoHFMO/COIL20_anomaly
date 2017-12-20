#!/usr/local/bin/python

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, RepeatVector, Lambda, Dropout, BatchNormalization, Flatten
from keras.models import Model, Sequential
from keras import regularizers
from keras import objectives
from keras.datasets import mnist
import numpy as np
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

from sklearn import svm

from keras import backend as K
from keras.models import load_model
import math
from sklearn import metrics
from collections import Counter
from keras.layers.core import Activation
from keras import optimizers

from sklearn.cross_validation import train_test_split
import scipy as sp
from keras.utils import np_utils

from scipy import stats

import constants

from sklearn.metrics import confusion_matrix

def pre_processing_svhn(x_train, y_train, tx_separate):
    
    '''
    :Obj: Este método junta duas classes e as separar conforme a taxa dada
    :param: x_train: Todos os dados do treinamento do dataset
    :param: y_train: Todos os labels do treinamento do dataset
    :return: X_train, y_train, X_test, y_test: Retorna o conjunto de treinamento e teste com seus respectivos labels
    '''

    # ajustando o formato dos dados
    x_train = np.transpose(x_train)

    b = []
    for i in range(len(x_train)):
        b.append(np.transpose(x_train[i]))
    print('fim')
    b = np.asarray(b)
    #b.shape
    x_train = b.copy()
    del b
    print(x_train.shape)


    # transformando os labels em um vetor
    list_y = []
    for i in range(len(y_train)):
        list_y.append(y_train[i][0])
    print('y_test')
    print(list_y)
    y_train = list_y.copy()
    y_train = np.asarray(y_train)
    
    # identificando a classe anormal
    digit_anomaly = int(constants.ANOMALY_CLASS)
        
    
    # colocando os dados em escala cinza
    x_train = x_train.astype('float32')
    x_train /= 255.
    
    # Selecionando os exemplos que são anormais 
    X_anomaly = x_train[y_train == digit_anomaly].copy()
    Y_anomaly = y_train[y_train == digit_anomaly].copy()
        
        
    # Selecionando os exemplos normais
    X_normal = x_train[y_train != digit_anomaly].copy()
    Y_normal = y_train[y_train != digit_anomaly].copy()
    
    
    # dividindo os dados em 80 treinamento e 20 teste
    X_train, X_test, y_train, y_test = train_test_split(X_normal, Y_normal, test_size=tx_separate, random_state=constants.RANDOM_STATE)
    
    # printando os dados de treinamento
    print('y_train values')
    print(list(y_train))
    print('y_test values')
    print(list(y_test))
    
    # Criando o conjunto de teste - juntando os dados de treinamento anormais com dados da validação normais 
    X_test = np.concatenate([X_test, X_anomaly])
    
    # Criando o conjunto de teste - juntando os labels de treinamento anormais com dados da validação normais 
    y_test = np.concatenate([y_test, Y_anomaly])
    
    
    print('X_test')
    print(X_test.shape)
    
    print('Y_test')
    print(y_test.shape)
    
    
    # printando a quantidade de exemplos para cada classe
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    y_train = np.transpose([y_train])
    y_test = np.transpose([y_test])
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    return X_train, y_train, X_test, y_test

def pre_processing_si(train, target, clf_type):
    
    '''
    :Obj: Este método separa a classe anormal para ser usada como teste e divide o dataset em treinamento e teste
    :param: train: Todos os dados de treinamento do dataset
    :param: target: Todos os labels do treinamento 
    :return: X_train, y_train, X_test, y_test: Retorna o conjunto de treinamento e teste com seus respectivos labels
    
    '''
    
    # transformando os labels em um vetor
    list_y = []
    for i in range(len(target)):
        list_y.append(target[i][0])
    print('y_test')
    print(list_y)
    target = list_y.copy()
    target = np.asarray(target)
    print("shape target: ", target.shape)
    
    # identificando a classe anormal
    digit = int(constants.ANOMALY_CLASS)
    
    # colocando os dados em escala cinza
    train = train.astype('float32')

    # Selecionando os exemplos que são anormais 
    X_digit = train[target == digit].copy()
    Y_digit = target[target == digit].copy()
    
    # variaveis para armazenar os dados normais
    X_not_digit, Y_not_digit = [], []
    
    '''
    # for para armazenar os dados normais
    train_class = [1, 4, 5]
    for i in range(len(target)):
        if(target[i] in train_class):
            X_not_digit.append(train[i])
            Y_not_digit.append(target[i])
    X_not_digit = np.asarray(X_not_digit)
    Y_not_digit = np.asarray(Y_not_digit)
    '''
    
    # for para treinar com todas as classes
    for i in range(len(target)):
        if(target[i] != digit):
            X_not_digit.append(train[i])
            Y_not_digit.append(target[i])
    X_not_digit = np.asarray(X_not_digit)
    Y_not_digit = np.asarray(Y_not_digit)
    
    
    # remodelando os dados caso seja um OCSVM
    if((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        X_not_digit = X_not_digit.reshape((len(X_not_digit), np.prod(X_not_digit.shape[1:])))
        X_digit = X_digit.reshape((len(X_digit), np.prod(X_digit.shape[1:])))
        
    else:
        # Remodelando os dados para entrar na rede
        X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
        X_digit = X_digit.reshape(X_digit.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)

    # Dividindo o dataset em conjunto de treinamento e teste.
    X_train, X_valid, y_train, y_valid = train_test_split(X_not_digit, Y_not_digit, test_size=0.20, random_state=constants.RANDOM_STATE)
    
    # printando os dados de treinamento
    print('y_train values')
    print(list(y_train))
    
    # printando os dados de teste
    print('y_valid values')
    print(list(y_valid))
    
    # Criando o conjunto de teste - juntando os dados de treinamento normais com dados da validação anormais 
    if((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        X_test = np.zeros((len(X_digit)+len(X_valid), len(X_digit[0])))
        X_test[:len(X_digit)] = X_digit
        X_test[len(X_digit):] = X_valid
    else:
        X_test = np.zeros((len(X_digit)+len(X_valid), len(X_digit[0]), len(X_digit[0]), 1))
        X_test[:len(X_digit)] = X_digit
        X_test[len(X_digit):] = X_valid
    
    # Remodelando os dados e ajustando para a forma apropriada
    if((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    else:
        X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    
    # Juntando os labels das saidas para o conjunto de teste
    y_test = np.append(Y_digit, y_valid)
    
    
    # printando os labels do conjunto de treinamento 
    print('---- Y_train -----')
    print(list(y_train))
    print('Len Y_train:' + str(len(y_train)) )
    
    
    # printando os labels do conjunto de teste 
    print('---- Y_TEST -----')
    print(list(y_test))
    print('Len Y_TEST:' + str(len(y_test)) )
    
    # printando a quantidade de exemplos para cada classe para o treinamento
    unique, counts = np.unique(y_train, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    # printando a quantidade de exemplos para cada classe para o teste
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    # retornando o conjunto de treinamento e teste
    return X_train, y_train, X_test, y_test

def pre_processing_coil20(train, target, clf_type):
    
    '''
    :Obj: Este método separa a classe anormal para ser usada como teste e divide o dataset em treinamento e teste
    :param: train: Todos os dados de treinamento do dataset
    :param: target: Todos os labels do treinamento 
    :return: X_train, y_train, X_test, y_test: Retorna o conjunto de treinamento e teste com seus respectivos labels
    
    '''
    
    # transformando os labels em um vetor
    #target = target.transpose()
    list_y = []
    for i in range(len(target)):
        list_y.append(target[i][0])
    print('y_test')
    print(list_y)
    target = list_y.copy()
    target = np.asarray(target)
    print("shape target: ", target.shape)
    
    # identificando a classe anormal
    digit = int(constants.ANOMALY_CLASS)
    
    # colocando os dados em escala cinza
    train = train.astype('float32')

    # Selecionando os exemplos que são anormais 
    X_digit = train[target == digit].copy()
    Y_digit = target[target == digit].copy()
    
    # variaveis para armazenar os dados normais
    X_not_digit, Y_not_digit = [], []
    
    '''
    # for para armazenar os dados normais
    train_class = [1, 4, 5]
    for i in range(len(target)):
        if(target[i] in train_class):
            X_not_digit.append(train[i])
            Y_not_digit.append(target[i])
    X_not_digit = np.asarray(X_not_digit)
    Y_not_digit = np.asarray(Y_not_digit)
    '''
    
    # for para treinar com todas as classes
    for i in range(len(target)):
        if(target[i] != digit):
            X_not_digit.append(train[i])
            Y_not_digit.append(target[i])
    X_not_digit = np.asarray(X_not_digit)
    Y_not_digit = np.asarray(Y_not_digit)
        
    # remodelando os dados caso seja um OCSVM
    if((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        X_not_digit = X_not_digit.reshape((len(X_not_digit), np.prod(X_not_digit.shape[1:])))
        X_digit = X_digit.reshape((len(X_digit), np.prod(X_digit.shape[1:])))
        
    else:
        # Remodelando os dados para entrar na rede
        X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.MNIST_IMG_CHANNELS)
        X_digit = X_digit.reshape(X_digit.shape[0], constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.MNIST_IMG_CHANNELS)
        
    # Dividindo o dataset em conjunto de treinamento e teste.
    X_train, X_valid, y_train, y_valid = train_test_split(X_not_digit, Y_not_digit, test_size=0.20, random_state=constants.RANDOM_STATE)
    
    # printando os dados de treinamento
    print('y_train values')
    print(list(y_train))
    
    # printando os dados de teste
    print('y_valid values')
    print(list(y_valid))
    
    # Criando o conjunto de teste - juntando os dados de treinamento normais com dados da validação anormais 
    if((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        X_test = np.zeros((len(X_digit)+len(X_valid), len(X_digit[0])))
        X_test[:len(X_digit)] = X_digit
        X_test[len(X_digit):] = X_valid
    else:
        X_test = np.zeros((len(X_digit)+len(X_valid), len(X_digit[0]), len(X_digit[0]), 1))
        X_test[:len(X_digit)] = X_digit
        X_test[len(X_digit):] = X_valid
    
    # Remodelando os dados e ajustando para a forma apropriada
    if((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
    else:
        X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], constants.CIFAR_IMG_ROWS, constants.CIFAR_IMG_COLS, constants.MNIST_IMG_CHANNELS)
        
    # Juntando os labels das saidas para o conjunto de teste
    y_test = np.append(Y_digit, y_valid)
    
    
    # printando os labels do conjunto de treinamento 
    print('---- Y_train -----')
    print(list(y_train))
    print('Len Y_train:' + str(len(y_train)) )
    
    
    # printando os labels do conjunto de teste 
    print('---- Y_TEST -----')
    print(list(y_test))
    print('Len Y_TEST:' + str(len(y_test)) )
    
    # printando a quantidade de exemplos para cada classe para o treinamento
    unique, counts = np.unique(y_train, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    # printando a quantidade de exemplos para cada classe para o teste
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    # retornando o conjunto de treinamento e teste
    return X_train, y_train, X_test, y_test

def concatenate_and_separate(x_train, y_train, tx_separate):
    
    '''
    :Obj: Este método junta duas classes e as separar conforme a taxa dada
    :param: x_train: Todos os dados do treinamento do dataset
    :param: y_train: Todos os labels do treinamento do dataset
    :return: X_train, y_train, X_test, y_test: Retorna o conjunto de treinamento e teste com seus respectivos labels
    '''

    # transformando os labels em um vetor
    list_y = []
    for i in range(len(y_train)):
        list_y.append(y_train[i][0])
    print('y_test')
    print(list_y)
    y_train = list_y.copy()
    y_train = np.asarray(y_train)
    
    # identificando a classe anormal
    digit_anomaly = int(constants.ANOMALY_CLASS)
        
    
    # colocando os dados em escala cinza
    x_train = x_train.astype('float32')
    x_train /= 255.
    
    # Selecionando os exemplos que são anormais 
    X_anomaly = x_train[y_train == digit_anomaly].copy()
    Y_anomaly = y_train[y_train == digit_anomaly].copy()
        
        
    # Selecionando os exemplos normais
    X_normal = x_train[y_train != digit_anomaly].copy()
    Y_normal = y_train[y_train != digit_anomaly].copy()
    
    
    # dividindo os dados em 80 treinamento e 20 teste
    X_train, X_test, y_train, y_test = train_test_split(X_normal, Y_normal, test_size=tx_separate, random_state=constants.RANDOM_STATE)
    
    # printando os dados de treinamento
    print('y_train values')
    print(list(y_train))
    print('y_test values')
    print(list(y_test))
    
    # Criando o conjunto de teste - juntando os dados de treinamento anormais com dados da validação normais 
    X_test = np.concatenate([X_test, X_anomaly])
    
    # Criando o conjunto de teste - juntando os labels de treinamento anormais com dados da validação normais 
    y_test = np.concatenate([y_test, Y_anomaly])
    
    
    print('X_test')
    print(X_test.shape)
    
    print('Y_test')
    print(y_test.shape)
    
    
    # printando a quantidade de exemplos para cada classe
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    y_train = np.transpose([y_train])
    y_test = np.transpose([y_test])
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    return X_train, y_train, X_test, y_test

def pre_processing_SVHN_data_without_class_CNN(train, target):
    '''
    :Obj: Este método separa a classe anormal para ser usada como teste e divide o dataset em treinamento e teste
    :param: train: Todos os dados de treinamento do dataset
    :param: target: Todos os dados de teste do dataset
    :return: X_train, y_train, X_test, y_test: Retorna o conjunto de treinamento e teste com seus respectivos labels
    
    '''
    #identificando a classe anormal
    digit = int(constants.ANOMALY_CLASS)
    
    # colocando os dados em escaal cinza
    train = train.astype('float32')
    train /= 255.

    # Selecionando os exemplos que são anormais 
    X_digit = train[target == digit].copy()
    Y_digit = target[target == digit].copy()
    
    
    # Selecionando os exemplos normais
    X_not_digit = train[target != digit].copy()
    Y_not_digit = target[target != digit].copy()
      
    # Remodelando os dados para entrar na rede
    X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    X_digit = X_digit.reshape(X_digit.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)

    # Dividindo o dataset em conjunto de treinamento e teste.
    X_train, X_valid, y_train, y_valid = train_test_split(X_not_digit, Y_not_digit, test_size=0.20, random_state=constants.RANDOM_STATE)
    
    #printando os dados de treinamento
    print('y_train values')
    print(list(y_train))
    print('y_valid values')
    print(list(y_valid))
    
    # Criando o conjunto de teste - juntando os dados de treinamento anormais com dados da validação normais 
    X_test = np.append(X_digit, X_valid)
    
    # Remodelando os dados e ajustando para a forma apropriada
    X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    
    # Juntando os labels das saidas para o conjunto de teste
    y_test = np.append(Y_digit, y_valid)
    
    # printando os labels do conjunto de teste 
    print('---- Y_TEST -----')
    print(list(y_test))
    print('Len Y_TEST:' + str(len(y_test)) )
    
    # printando a quantidade de exemplos para cada classe
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    # retornando o conjunto de treinamento e teste
    return X_train, y_train, X_test, y_test

def pre_processing_MNIST_data_without_class_CNN(train, target):
    '''
    :Obj: Este método separa a classe anormal para ser usada como teste e divide o dataset em treinamento e teste
    :param: train: Todos os dados de treinamento do dataset
    :param: target: Todos os labels do treinamento 
    :return: X_train, y_train, X_test, y_test: Retorna o conjunto de treinamento e teste com seus respectivos labels
    
    '''
    # identificando a classe anormal
    digit = int(constants.ANOMALY_CLASS)
    
    # colocando os dados em escala cinza
    train = train.astype('float32')
    train /= 255.

    # Selecionando os exemplos que são anormais 
    X_digit = train[target == digit].copy()
    Y_digit = target[target == digit].copy()
    
    
    # Selecionando os exemplos normais
    X_not_digit = train[target != digit].copy()
    Y_not_digit = target[target != digit].copy()
      
    # Remodelando os dados para entrar na rede
    X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    X_digit = X_digit.reshape(X_digit.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)

    # Dividindo o dataset em conjunto de treinamento e teste.
    X_train, X_valid, y_train, y_valid = train_test_split(X_not_digit, Y_not_digit, test_size=0.20, random_state=constants.RANDOM_STATE)
    
    # printando os dados de treinamento
    print('y_train values')
    print(list(y_train))
    print('y_valid values')
    print(list(y_valid))
    
    # Criando o conjunto de teste - juntando os dados de treinamento anormais com dados da validação normais 
    X_test = np.append(X_digit, X_valid)
    
    # Remodelando os dados e ajustando para a forma apropriada
    X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    
    # Juntando os labels das saidas para o conjunto de teste
    y_test = np.append(Y_digit, y_valid)
    
    # printando os labels do conjunto de teste 
    print('---- Y_TEST -----')
    print(list(y_test))
    print('Len Y_TEST:' + str(len(y_test)) )
    
    # printando a quantidade de exemplos para cada classe
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    # retornando o conjunto de treinamento e teste
    return X_train, y_train, X_test, y_test

def pre_processing_MNIST_data_without_class_CNN2(train, target):
    '''
    :Obj: Este método separa a classe anormal para ser usada como teste e divide o dataset em treinamento e teste
    :param: train: Todos os dados de treinamento do dataset
    :param: target: Todos os labels do treinamento 
    :return: X_train, y_train, X_test, y_test: Retorna o conjunto de treinamento e teste com seus respectivos labels
    
    '''
    
    # classe para ser treinada
    train_class = [7, 8, 9]
    
    # identificando a classe anormal
    digit = int(constants.ANOMALY_CLASS)
    
    # colocando os dados em escala cinza
    train = train.astype('float32')
    train /= 255.

    # Selecionando os exemplos que são anormais 
    X_digit = train[target == digit].copy()
    Y_digit = target[target == digit].copy()
    
    # variaveis para armazenar os dados normais
    X_not_digit, Y_not_digit = [], []
    
    # for para armazenar os dados normais
    for i in range(len(target)):
        if(target[i] in train_class):
            X_not_digit.append(train[i])
            Y_not_digit.append(target[i])
            
    X_not_digit = np.asarray(X_not_digit)
    Y_not_digit = np.asarray(Y_not_digit)
        
    # Remodelando os dados para entrar na rede
    X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    X_digit = X_digit.reshape(X_digit.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)

    # Dividindo o dataset em conjunto de treinamento e teste.
    X_train, X_valid, y_train, y_valid = train_test_split(X_not_digit, Y_not_digit, test_size=0.20, random_state=constants.RANDOM_STATE)
    
    # printando os dados de treinamento
    print('y_train values')
    print(list(y_train))
    print('y_valid values')
    print(list(y_valid))
    
    # Criando o conjunto de teste - juntando os dados de treinamento anormais com dados da validação normais 
    X_test = np.append(X_digit, X_valid)
    
    # Remodelando os dados e ajustando para a forma apropriada
    X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], constants.MNIST_IMG_ROWS, constants.MNIST_IMG_COLS, constants.MNIST_IMG_CHANNELS)
    
    # Juntando os labels das saidas para o conjunto de teste
    y_test = np.append(Y_digit, y_valid)
    
    # printando os labels do conjunto de teste 
    print('---- Y_TEST -----')
    print(list(y_test))
    print('Len Y_TEST:' + str(len(y_test)) )
    
    # printando a quantidade de exemplos para cada classe para o treinamento
    unique, counts = np.unique(y_valid, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    # printando a quantidade de exemplos para cada classe para o teste
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
    
    # retornando o conjunto de treinamento e teste
    return X_train, y_train, X_test, y_test

def pre_processing_MNIST_data_without_class_OCSVM(train, target):
    digit = int(constants.ANOMALY_CLASS)
        
    train = train.astype('float32')
    train /= 255.

    # Selecting the examples where the output is ANOMALY_CLASS
    X_digit = train[target == digit].copy()
    Y_digit = target[target == digit].copy()
    
    # Selecting the examples where the output is not ANOMALY_CLASS
    X_not_digit = train[target != digit].copy()
    Y_not_digit = target[target != digit].copy()
      
    # Reshaping the X data to appropriate form
    X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)
    X_digit = X_digit.reshape(X_digit.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)

    #Spliting train data into training and test sets.
    X_train, X_valid, y_train, y_valid = train_test_split(X_not_digit, Y_not_digit, test_size=0.20, random_state=constants.RANDOM_STATE)
      
    print('y_train values')
    print(list(y_train))
    print('y_valid values')
    print(list(y_valid))
    
    # Appending the data with output as digit and data with output as != digit
    X_test = np.append(X_digit, X_valid)
    
    # Reshaping the appended data to appropriate form
#     X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], 1, IMG_ROWS, IMG_COLS)
    X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)
    
    # Appending the labels
    y_test = np.append(Y_digit, y_valid)
     
    print('---- Y_TEST -----')
    print(list(y_test))
    
#     return X_train, y_train, X_test, y_test
    ''' AQUI '''
#     return X_train, y_train, X_test, y_test
    
    X_train_temp = X_train[5000:10000]
    y_train_temp = y_train[5000:10000]
#     X_test_temp = X_test[5000:10000]
#     y_test_temp = y_test[5000:10000]
     
     
    unique, counts = np.unique(y_train_temp, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_train_temp')
    print(temp)
    
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test_temp')
    print(temp)
      
    return X_train_temp, y_train_temp, X_test, y_test

def pre_processing_MNIST_data_OCSVM(train, target):
    digit = int(constants.ANOMALY_CLASS)
        
    train = train.astype('float32')
    train /= 255.

    # Selecting the examples where the output is ANOMALY_CLASS
    X_digit = train[target == digit].copy()
    Y_digit = target[target == digit].copy()
    
    # Selecting the examples where the output is not ANOMALY_CLASS
    X_not_digit = train[target != digit].copy()
    Y_not_digit = target[target != digit].copy()
      
    # Reshaping the X data to appropriate form
    X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)
    X_digit = X_digit.reshape(X_digit.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)
    
    # All the train data are normal_data (Inliers -> representation: 1)
    z_valid = np.ones((Y_not_digit.shape[0],), dtype=np.int)

    #Spliting train data into training and test sets.
    X_train, X_valid, y_train, y_valid = train_test_split(X_not_digit, z_valid, test_size=0.20, random_state=constants.RANDOM_STATE)
      
    print('y_train values')
    print(list(y_train))
    print('y_valid values')
    print(list(y_valid))
    
    # Appending the data with output as digit and data with output as != digit
    X_test = np.append(X_digit, X_valid)
    
    # Reshaping the appended data to appropriate form
#     X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], 1, IMG_ROWS, IMG_COLS)
    X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)
    
    # anormal_data (Outliers -> representation: 0)
    z_digit = np.zeros((Y_digit.shape[0],), dtype=np.int)
#     z_digit = z_digit * (-1)


    # Appending the labels
#     y_test = np.append(Y_digit, y_valid)
    y_test = np.append(z_digit, y_valid)
     
    print('---- Y_TEST -----')
    print(list(y_test))
    
#     return X_train, y_train, X_test, y_test
    ''' AQUI '''
#     return X_train, y_train, X_test, y_test
    
    X_train_temp = X_train[5000:10000]
    y_train_temp = y_train[5000:10000]
#     X_test_temp = X_test[5000:10000]
#     y_test_temp = y_test[5000:10000]
     
    unique, counts = np.unique(y_train_temp, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_train_temp')
    print(temp)
    
    unique, counts = np.unique(y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test_temp')
    print(temp)
      
    return X_train_temp, y_train_temp, X_test, y_test

def pre_processing_MNIST_data_one_class(train, target):
    ''' OCSVM: Inliers are labeled 1, while outliers are labeled -1. '''
    digit = int(constants.ANOMALY_CLASS)
    
    train = train.astype('float32')
    train /= 255.

    # Selecting the examples where the output is 0
    X_digit = train[target == digit].copy()
    Y_digit = target[target == digit].copy()
    
    # Selecting the examples where the output is not 0
    X_not_digit = train[target != digit].copy()
    Y_not_digit = target[target != digit].copy()
      
#     if K.image_dim_ordering() == 'th':
    X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)
    X_digit = X_digit.reshape(X_digit.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)
#         X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#         input_shape = (1, img_rows, img_cols)
#     else:
#         X_not_digit = X_not_digit.reshape(X_not_digit.shape[0], IMG_ROWS * IMG_COLS)
#         X_digit = X_digit.reshape(X_digit.shape[0], IMG_ROWS * IMG_COLS)
#         X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#         input_shape = (img_rows, img_cols, 1)
    
#     print('X_train shape:', train.shape)
#     print(X_not_digit.shape[0], 'not_digit samples')
#     print(X_not_digit.shape, 'not_digit shape')
#     print(X_digit.shape[0], 'digit samples')
#     print(X_digit.shape, 'digit shape')
    
#     print('*************************')
#     print(X_not_digit[0])
    
#     z_valid = np.zeros((Y_not_digit.shape[0],), dtype=np.int)
    # All the train data are normal_data (Inliers -> representation: 1)
    z_valid = np.ones((Y_not_digit.shape[0],), dtype=np.int)
    
    #Spliting train data into training and test sets.
    X_train, X_valid, y_train, y_valid = train_test_split(X_not_digit, z_valid, test_size=0.20, random_state=constants.RANDOM_STATE)
    
#     print('train_test_split: x')
#     print(X_train.shape[0], 'train samples')
#     print(X_train.shape, 'train shape')
#     print(X_valid.shape[0], 'valid samples')
#     print(X_valid.shape, 'valid shape')
#     
#     print('train_test_split: y')
#     print(y_train.shape[0], 'train samples')
#     print(y_train.shape, 'train shape')
#     print(y_valid.shape[0], 'valid samples')
#     print(y_valid.shape, 'valid shape')
#     
    print('y_train values')
    print(list(y_train))
    print('y_valid values')
    print(list(y_valid))
    
#     print('y_train values shape')
#     print(y_train.shape)
#     print('y_valid values shape')
#     print(y_valid.shape)
    
    # Appending the data with output as digit and data with output as != digit
    X_test = np.append(X_digit, X_valid)
    
    # Reshaping the appended data to appropriate form
    X_test = X_test.reshape(X_digit.shape[0] + X_valid.shape[0], constants.MNIST_IMG_ROWS * constants.MNIST_IMG_COLS)
    
    # anormal_data (Outliers -> representation: -1)
    z_digit = np.ones((Y_digit.shape[0],), dtype=np.int)
    z_digit = z_digit * (-1)
    
    # Appending the labels
    y_test = np.append(z_digit, y_valid)
    
    print('y_test values')
    print(list(y_test))

#     Y_train = Y_labels == digit 
#     Y_train = Y_train.astype(int)
#     print('TEST')
#     print(X_test.shape[0], 'train samples')
#     print(X_test.shape, 'train shape')
#     print(y_test.shape[0], 'valid samples')
#     print(y_test.shape, 'valid shape')
#     
#     print('---- Y_TEST -----')
#     print(list(y_test))
    ''' AQUI '''
#     return X_train, y_train, X_test, y_test
    
    X_train_temp = X_train[5000:10000]
    y_train_temp = y_train[5000:10000]
    X_test_temp = X_test[5000:10000]
    y_test_temp = y_test[5000:10000]
     
    print('temp')
    print(list(y_test_temp))
    
    unique, counts = np.unique(y_test_temp, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test_temp')
    print(temp)
      
    return X_train_temp, y_train_temp, X_test_temp, y_test_temp

def euclidean_distance(A, B):
    return np.linalg.norm(A - B)

def calc_error_metrics(error):
    er_mean = np.mean(error)
    er_min = np.min(error)
    er_max = np.max(error)
    
    first_quartile = np.percentile(error, 25) # first quartile
    second_quartile = np.percentile(error, 50) # second quartile = median
    third_quartile = np.percentile(error, 75) # third quartile
    
    print("er_mean: ")
    print(er_mean)
    
    print("er_min: ")
    print(er_min)
    
    print("er_max: ")
    print(er_max)
    
    print("first_quartile: ")
    print(first_quartile)
    
    print("second_quartile: ")
    print(second_quartile)
    
    print("third_quartile: ")
    print(third_quartile)
        
    return er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll    

def gridsearch_error_logloss(real_data, decoded_data):
    '''
    :obj: Método que computa o erro de reconstruçao para os dados trasformados pela rede
    :param: real_data: Dados reais
    :param: decoded_data: Dados decodificados
    :return: er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile: Retorna o erro de reconstrução para o dataset
    
    '''
    
    error = []
    for i in range(decoded_data.shape[0]):     
        err = logloss(real_data[i], decoded_data[i])
        err_sum = np.sum(err)
        if(math.isnan(err_sum) or math.isinf(err_sum)):
            continue
        error.append(err_sum)
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = calc_error_metrics(error)
    return er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile

def gridsearch_error_logloss_debug(real_data, decoded_data):
    error = []
    for i in range(decoded_data.shape[0]):     
        err = logloss(real_data[i], decoded_data[i])
        #err_sum = np.sum(err)
        err_sum = 0
        for j in range(len(err)):
            if(math.isnan(err[j][0]) or math.isinf(err[j][0])):
                continue
            else:
                err_sum += err[j][0]
        #print("[", i, "]: ", err_sum)
        error.append(err_sum)
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = calc_error_metrics(error)
    return er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile, error

def error_logloss(real_data, decoded_data):
    error = []
    for i in range(decoded_data.shape[0]):     
        err = logloss(real_data[i], decoded_data[i])
        err_sum = np.sum(err)
        if(math.isnan(err_sum) or math.isinf(err_sum)):
            continue
        error.append(err_sum)
    return error

def gridsearch_error_mse(real_data, decoded_data):
    error = []
    for i in range(decoded_data.shape[0]):        
#         err = metrics.mean_squared_error(real_data[i], decoded_data[i])
        err = ((real_data[i] - decoded_data[i]) ** 2).mean(axis=None)   
        if(math.isnan(err) or math.isinf(err)):
            continue
#         if i == 39152:
#             print('Index: ' + str(i))
#         print('Index: ' + str(i))
#         print(err)
        error.append(err)
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = calc_error_metrics(error)
    return er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile

def anomaly_detection_mse(real_data, decoded_data, threshold):
    anomaly_data = []
    print('threshold ' + str(threshold))
    for i in range(decoded_data.shape[0]):        
#         err = metrics.mean_squared_error(real_data[i], decoded_data[i])
        err = ((real_data[i] - decoded_data[i]) ** 2).mean(axis=None)
#         print('Index: ' + str(i))
#         print(err)
        if(err > threshold): # mse
            anomaly_data.append(constants.ANOMALY_DATA_REPRESENTATION) # anomaly
        else:
            anomaly_data.append(constants.NORMAL_DATA_REPRESENTATION) # normal
    ''' anomaly_data: anomaly = ANOMALY_DATA_REPRESENTATION / normal = NORMAL_DATA_REPRESENTATION'''
    anomaly_data = np.asarray(anomaly_data)
    print("anomaly_data: ")
    print(anomaly_data)
    return anomaly_data

def anomaly_detection_logloss(real_data, decoded_data, threshold):
    anomaly_data = []
    print('Threshold: ' + str(threshold))
    for i in range(decoded_data.shape[0]):        
        err = logloss(real_data[i], decoded_data[i])
        err_sum = np.sum(err)
#         print('Index: ' + str(i))
#         print(err_sum)
        if(err_sum > threshold): # logloss
            anomaly_data.append(constants.ANOMALY_DATA_REPRESENTATION) # anomaly
        else:
            anomaly_data.append(constants.NORMAL_DATA_REPRESENTATION) # normal
    ''' anomaly_data: anomaly = ANOMALY_DATA_REPRESENTATION / normal = NORMAL_DATA_REPRESENTATION'''
    print('-- anomaly_list -- ')
    print(anomaly_data)
    print('LEN anomaly_data:' + str(len(anomaly_data)))
    anomaly_data = np.asarray(anomaly_data)
    print("anomaly_data: ")
    print(anomaly_data)
    return anomaly_data

def confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def anomaly_detection_confidence_interval(real_data, decoded_data, lower_limit, upper_limit):
    anomaly_data = []
    print('Confidence Interval: [' + str(lower_limit) + ' , ' + str(upper_limit) +']')
    for i in range(decoded_data.shape[0]):        
        err = logloss(real_data[i], decoded_data[i])
        err_sum = np.sum(err)
#         print('Index: ' + str(i))
#         print(err_sum)
        if((err_sum > lower_limit) and (err_sum < upper_limit)): # logloss
            anomaly_data.append(constants.ANOMALY_DATA_REPRESENTATION) # anomaly
        else:
            anomaly_data.append(constants.NORMAL_DATA_REPRESENTATION) # normal
    ''' anomaly_data: anomaly = ANOMALY_DATA_REPRESENTATION / normal = NORMAL_DATA_REPRESENTATION'''
    print('-- anomaly_list -- ')
    print(anomaly_data)
    anomaly_data = np.asarray(anomaly_data)
#     print("anomaly_data: ")
#     print(anomaly_data)
    return anomaly_data

def anomaly_detection_rate(y_test,test_anomaly_data_indexes, anomaly_data_indexes):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    
    for i in range(len(y_test)):
        if((i in test_anomaly_data_indexes) and (i in anomaly_data_indexes)):
            true_positive = true_positive + 1
        elif((i not in test_anomaly_data_indexes) and (i in anomaly_data_indexes)):
            false_positive = false_positive + 1
        elif((i not in test_anomaly_data_indexes) and (i not in anomaly_data_indexes)):
            true_negative = true_negative + 1
        elif((i in test_anomaly_data_indexes) and (i not in anomaly_data_indexes)):
            false_negative = false_negative + 1
    return true_positive, false_positive, true_negative, false_negative 

def mcc_coef(true_positive, false_positive, true_negative, false_negative):
    num = (true_positive * true_negative) - (false_positive * false_negative)
    den = math.sqrt((true_positive + false_positive)*(true_positive + false_negative)*(true_negative + false_positive)*(true_negative + false_negative))
    mcc = num/den
    return mcc

def detection_rate(true_positive, nb_anomaly):
    detection = (true_positive / float(nb_anomaly)) 
    return detection

def false_alarm_rate(false_positive, nb_normal_data):
    false_alarm = (false_positive / float(nb_normal_data)) 
    return false_alarm

def calc_auc_roc(y_true, y_pred):
    ''' Area under the curve of the receiver operating characteristic (AUC ROC) '''
    ''' y_true and y_pred: np.array of 1 (normal_data) and 0 (anomaly_data)'''
    auc_roc = metrics.roc_auc_score(y_true, y_pred)
    return auc_roc    

def calc_auc_prc(y_true, y_pred):
    ''' Area under the curve of the precision recall curve (AUC PRC) '''
    ''' y_true and y_pred: np.array of 1 (normal_data) and 0 (anomaly_data)'''
#     precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
#     auc_prc = metrics.auc(recall, precision)
    auc_prc = metrics.average_precision_score(y_true, y_pred)
    return auc_prc

def calc_f1_score(y_true, y_pred):
    ''' y_true and y_pred: np.array of 1 (normal_data) and 0 (anomaly_data)'''
    score = metrics.f1_score(y_true, y_pred, average='binary')
    return score

def calc_fscore_support(y_true, y_pred):
    ''' y_true and y_pred: np.array of 1 (normal_data) and 0 (anomaly_data)'''
#     score = metrics.f1_score(y_true, y_pred, average='binary')
    score = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
    return score

# def calc_precision(true_positive, false_positive):
#     ''' The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. '''
#     precision = (true_positive / float(true_positive + false_positive))
#     return precision
# 
# def calc_recall(true_positive, false_negative):
#     ''' The recall is intuitively the ability of the classifier to find all the positive samples. '''
#     recall = (true_positive / float(true_positive + false_negative))
#     return recall

# def print_information(y_test, y_pred):    
#     anomaly_list = list(y_pred)
#     anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == ANOMALY_DATA_REPRESENTATION]
#     normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x != ANOMALY_DATA_REPRESENTATION]
#     
#     y_true = []
#     for i in y_test:
#         if(i == ANOMALY_DIGIT):
#             y_true.append(ANOMALY_DATA_REPRESENTATION) # anomaly
#         else:
#             y_true.append(NORMAL_DATA_REPRESENTATION) # normal
# #     y_true = np.asarray(y_true)
# #     
# #     y_test_list = list(y_test)
#     ''' anomaly = ANOMALY_DATA_REPRESENTATION / normal = NORMAL_DATA_REPRESENTATION '''
#     test_anomaly_data_indexes = [i for i,x in enumerate(y_true) if x == ANOMALY_DATA_REPRESENTATION]
#     test_normal_data_indexes = [i for i,x in enumerate(y_true) if x != ANOMALY_DATA_REPRESENTATION]
#     y_true = np.asarray(y_true)
#      
#     ''' Computing anomaly_detection_rate'''
#     true_positive, false_positive, true_negative, false_negative = anomaly_detection_rate(y_test, test_anomaly_data_indexes, anomaly_data_indexes)
#     
#     print('size anomaly_data_indexes')
#     print(len(anomaly_data_indexes))
#     
#     print('size normal_data_indexes')
#     print(len(normal_data_indexes))
#     
#     print('true_positive')
#     print(true_positive)
#     
#     print('false_positive')
#     print(false_positive)
#     
#     print('true_negative')
#     print(true_negative)
#     
#     print('false_negative')
#     print(false_negative)
#        
#     print('detection_rate')
#     det_rate = detection_rate(true_positive, len(test_anomaly_data_indexes))
#     print(det_rate)
#     
#     print('false_alarm_rate')
#     false_rate = false_alarm_rate(false_positive, len(test_normal_data_indexes))
#     print(false_rate)
#     
#     print('precision_rate')
#     prec = calc_precision(true_positive, false_positive)
#     print(prec)
#     
#     print('recall_rate')
#     rec = calc_recall(true_positive, false_negative)
#     print(rec)
#     
#     print('mathews_corrcoef')
#     mcc = mcc_coef(true_positive, false_positive, true_negative, false_negative)
#     print(mcc)
#     
#     print('Area under the curve of the receiver operating characteristic (AUC ROC)')
#     auc_roc = calc_auc_roc(y_true, y_pred)
#     print(auc_roc)
#      
#     print('Area under the curve of the precision recall curve (AUC PRC)')
#     auc_prc = calc_auc_prc(y_true, y_pred)
#     print(auc_prc)
#      
#     print('F1 score')
#     f_score = calc_f1_score(y_true, y_pred)
#     print(f_score)
#     
#     print('fscore support')
#     f_score_support = calc_fscore_support(y_true, y_pred)
#     print(f_score_support)
#     
#     print('\nALPHA: '+ str(ALPHA))

# def print_information(y_test, y_pred):  
#     anomaly_list = list(y_pred)
#     anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == ANOMALY_DATA_REPRESENTATION]
#     normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x != ANOMALY_DATA_REPRESENTATION]
#       
#     y_true = []
#     for i in y_test:
#         if(i == ANOMALY_CLASS):
#             y_true.append(ANOMALY_DATA_REPRESENTATION) # anomaly
#         else:
#             y_true.append(NORMAL_DATA_REPRESENTATION) # normal
# 
#     ''' anomaly = ANOMALY_DATA_REPRESENTATION / normal = NORMAL_DATA_REPRESENTATION '''
#     test_anomaly_data_indexes = [i for i,x in enumerate(y_true) if x == ANOMALY_DATA_REPRESENTATION]
#     test_normal_data_indexes = [i for i,x in enumerate(y_true) if x != ANOMALY_DATA_REPRESENTATION]
#     y_true = np.asarray(y_true)
#     
#     print('size anomaly_data_indexes')
#     print(len(anomaly_data_indexes))
#      
#     print('size normal_data_indexes')
#     print(len(normal_data_indexes))
#     
#     print('size anomaly_data_indexes Test')
#     print(len(test_anomaly_data_indexes))
#      
#     print('size normal_data_indexes Test')
#     print(len(test_normal_data_indexes))
#      
#     print('precision_score')
#     prec = metrics.precision_score(y_true, y_pred)
#     print(prec)
#     
#     print('recall_score')
#     rec = metrics.recall_score(y_true, y_pred)
#     print(rec)
#     
#     print('mathews_corrcoef')
#     mcc = metrics.matthews_corrcoef(y_true, y_pred)
#     print(mcc)
#     
#     print('Area under the curve of the receiver operating characteristic (AUC ROC)')
#     auc_roc = metrics.roc_auc_score(y_true, y_pred)
#     print(auc_roc)
#      
#     precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
#     area = metrics.auc(recall, precision)
#     print "Area Under PR Curve(AP): " + str(area) 
#      
#     print('Area under the curve of the precision recall curve (AUC PRC)')
#     auc_prc = metrics.average_precision_score(y_true, y_pred)
#     print(auc_prc)
#      
#     print('F1 score')
#     f_score = metrics.f1_score(y_true, y_pred, average='binary')
#     print(f_score)
#     
#     print('fscore support')
#     f_score_support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
#     print(f_score_support)
#     
#     return mcc, auc_roc, auc_prc, f_score

def anomaly_detection_cnn_threshold(max_indexes, max_values, threshold):
    cnn_threshold = []
    for i in range(len(max_values)):
        if(max_values[i] < threshold):
            cnn_threshold.append(constants.ANOMALY_CLASS)
        else:
            cnn_threshold.append(max_indexes[i])
    return cnn_threshold

def print_information(y_test, y_pred): 
    print('\nAnomaly class: ' + str(constants.ANOMALY_CLASS))
    y_true = y_test   
    anomaly_list = list(y_pred)
#     print('anomaly_list')
#     print(anomaly_list)
    anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == constants.ANOMALY_DATA_REPRESENTATION]
    normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x != constants.ANOMALY_DATA_REPRESENTATION]
      
    y_test_list = list(y_test)
    ''' anomaly = ANOMALY_DATA_REPRESENTATION / normal = NORMAL_DATA_REPRESENTATION '''
    test_anomaly_data_indexes = [i for i,x in enumerate(y_test_list) if x == constants.ANOMALY_DATA_REPRESENTATION]
    test_normal_data_indexes = [i for i,x in enumerate(y_test_list) if x != constants.ANOMALY_DATA_REPRESENTATION]
    
    print('size anomaly_data_indexes')
    print(len(anomaly_data_indexes))
     
    print('size normal_data_indexes')
    print(len(normal_data_indexes))
    
    print('size anomaly_data_indexes Test')
    print(len(test_anomaly_data_indexes))
     
    print('size normal_data_indexes Test')
    print(len(test_normal_data_indexes))
    
    cm = confusion_matrix(y_true, y_pred)
    print('\nconfusion_matrix')
    print(cm)
    
    ''' tn, fp, fn, tp '''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
     
    print('true negative')
    print(tn)
    print('false positive')
    print(fp)
    print('false negative')
    print(fn)
    print('true positive')
    print(tp)
    
#     nb_anomaly = float(len(test_anomaly_data_indexes))
#     det_rate = detection_rate(tp, nb_anomaly)
#     print('detection_rate')
#     print(det_rate)
     
    precision = (tp/float(tp +fp))
    recall = (tp/float(tp+fn))
     
    print('precision')
    print(precision)
    print('recall')
    print(recall)
        
    print('precision_score')
    prec = metrics.precision_score(y_true, y_pred)
    print(prec)
    
    print('recall_score')
    rec = metrics.recall_score(y_true, y_pred)
    print(rec)
    
    print('mathews_corrcoef')
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    print(mcc)
    
    print('Area under the curve of the receiver operating characteristic (AUC ROC)')
    auc_roc = metrics.roc_auc_score(y_true, y_pred)
    print(auc_roc)
    
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    area = metrics.auc(recall, precision)
    print("Area Under PR Curve(AP): " + str(area))  
     
    print('Area under the curve of the precision recall curve (AUC PRC)')
    auc_prc = metrics.average_precision_score(y_true, y_pred)
    print(auc_prc)
     
    print('F1 score')
    f_score = metrics.f1_score(y_true, y_pred, average='binary')
    print(f_score)
    
    print('fscore support')
    f_score_support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f_score_support)
    
    return mcc, auc_roc, auc_prc, f_score

def print_information_temp(y_test, y_pred):  
    anomaly_list = list(y_pred)
    anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == constants.ANOMALY_DATA_REPRESENTATION]
    normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x != constants.ANOMALY_DATA_REPRESENTATION]
      
    y_true = []
    for i in y_test:
        if(i == constants.ANOMALY_CLASS):
            y_true.append(constants.ANOMALY_DATA_REPRESENTATION) # anomaly
        else:
            y_true.append(constants.NORMAL_DATA_REPRESENTATION) # normal

    ''' anomaly = ANOMALY_DATA_REPRESENTATION / normal = NORMAL_DATA_REPRESENTATION '''
    test_anomaly_data_indexes = [i for i,x in enumerate(y_true) if x == constants.ANOMALY_DATA_REPRESENTATION]
    test_normal_data_indexes = [i for i,x in enumerate(y_true) if x != constants.ANOMALY_DATA_REPRESENTATION]
    y_true = np.asarray(y_true)
    
    print('size anomaly_data_indexes')
    print(len(anomaly_data_indexes))
     
    print('size normal_data_indexes')
    print(len(normal_data_indexes))
    
    print('size anomaly_data_indexes Test')
    print(len(test_anomaly_data_indexes))
     
    print('size normal_data_indexes Test')
    print(len(test_normal_data_indexes))
     
    print('precision_score')
    prec = metrics.precision_score(y_true, y_pred)
    print(prec)
    
    print('recall_score')
    rec = metrics.recall_score(y_true, y_pred)
    print(rec)
    
    print('mathews_corrcoef')
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    print(mcc)
    
    print('Area under the curve of the receiver operating characteristic (AUC ROC)')
    auc_roc = metrics.roc_auc_score(y_true, y_pred)
    print(auc_roc)
     
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    area = metrics.auc(recall, precision)
    print("Area Under PR Curve(AP): " + str(area)) 
     
    print('Area under the curve of the precision recall curve (AUC PRC)')
    auc_prc = metrics.average_precision_score(y_true, y_pred)
    print(auc_prc)
     
    print('F1 score')
    f_score = metrics.f1_score(y_true, y_pred, average='binary')
    print(f_score)
    
    print('fscore support')
    f_score_support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f_score_support)
    
    return mcc, auc_roc, auc_prc, f_score

def print_lists(list_mcc, list_auc_roc, list_auc_prc, list_f_score, list_alpha):
    print("\n--- LISTS ---")
    print("MCC")
    print(list_mcc)
    print("AUC_ROC")
    print(list_auc_roc)
    print("AUC_PRC")
    print(list_auc_prc)
    print("F_SCORE")
    print(list_f_score)
    print("ALPHA")
    print(list_alpha) 

def pre_processing_CIFAR_data_without_class(train, target, anomaly_class, clf_type): 
    '''
    :obj: retirar os objetos anomalos do conjunto de treinamento
    :param train: dados de treinamento 
    :param target: label dos dados de treinamento
    :param anomaly_class: label da classe anomala
    :param clf_type: tipo da rede neural usada
    :return X_normal, Y_normal: retorna o conjunto de treinamento somente com os dados normais
    '''
    
    if ((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        
        # remodelando os dados de treinamento
        train = train.reshape((len(train), np.prod(train.shape[1:])))

    
    # printando o novo formato    
    print('train')
    print(train.shape)

    # criando uma lista com os labels de treinamento
    anomaly_list = list(target)
    
    # salvando os indexes dos dados anomalos
    anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == anomaly_class]
    
    # salvando os indexes dos dados normais
    normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x != anomaly_class]
    
    
    # printando a quantidade de dados anomalos
    print('--- Anomaly_data_indexes ---')
    #print(anomaly_data_indexes)
    print(len(anomaly_data_indexes))
    
    
    # printando a quantidade de dados normais
    print('--- Normal_data_indexes ---')
    #print(normal_data_indexes)
    print(len(normal_data_indexes))
    
    
    # criando as variaveis para armazenar os dados e os labels dos dados normais e anormais
    X_normal_class, X_anomaly_class = [], []
    Y_normal_class, Y_anomaly_class = [], []
    
    # copiando os dados e os labels do conjunto de treinamento
    X_train = train.copy()
    Y_train = target.copy()
    
    # retirando os dados anormais do conjunto de treinamento
    for i in range(len(X_train)):
        if(i in anomaly_data_indexes):
            X_anomaly_class.append(X_train[i])
            Y_anomaly_class.append(Y_train[i])
        else:
            X_normal_class.append(X_train[i])
            Y_normal_class.append(Y_train[i])
    
    # transformando as variaveis em um array    
    X_normal = np.asarray(X_normal_class)
    X_anomaly = np.asarray(X_anomaly_class)
    Y_normal = np.asarray(Y_normal_class)
    Y_anomaly = np.asarray(Y_anomaly_class)
    
    
    # printando o formato dos dados das variaveis
    print('x_normal')
    print(X_normal.shape)
    print('x_anomaly')
    print(X_anomaly.shape)
    
    
    # printando o formato dos labels das variaveis
    print('y_normal')
    print(Y_normal.shape)
    print('y_anomaly')
    print(Y_anomaly.shape)
    
    
    # passa para o método os labels dos dados normais e recebe quem são os dados unicos e a quantidade de cada um
    unique, counts = np.unique(Y_normal, return_counts=True)
    
    # criando um dicionario com os dados separados
    temp = dict(zip(unique, counts))
    
    # printando o dicionario
    print('counter_Y_normal')
    print(temp)
    
    
    if ((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        X_train, X_valid, y_train, y_valid = train_test_split(X_normal, Y_normal, test_size=0.20, random_state=constants.RANDOM_STATE)
        
        X_train_temp = X_train[5000:7000]
        y_train_temp = y_train[5000:7000]
        
#         X_train_temp = X_train[5000:10000]
#         y_train_temp = y_train[5000:10000]
        
    #     print('temp')
    #     print(list(y_test))
        
        unique, counts = np.unique(y_train_temp, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_y_train_temp')
        print(temp)
        
        return X_train_temp, y_train_temp
    else:
        return X_normal, Y_normal

def pre_processing_CIFAR_data_without_class2(x_train, y_train, x_test, y_test, anomaly_class, clf_type): 
    '''
    :obj: retirar os objetos anomalos do conjunto de treinamento
    :param x_train: dados de treinamento 
    :param y_train: label dos dados de treinamento
    :param x_test: dados de teste 
    :param y_test: label dos dados de teste
    :param anomaly_class: label da classe anomala
    :param clf_type: tipo da rede neural usada
    :return X_normal, Y_normal: retorna o conjunto de treinamento somente com os dados normais
    '''
    ################################################## etapa dos dados de treinamento##################################################
    
    # classe para ser treinada
    train_class = [7, 8, 9]
    
    if ((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
        
        # remodelando os dados de treinamento
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    
    # printando o novo formato    
    print('x_train')
    print(x_train.shape)

    # criando uma lista com os labels de treinamento
    anomaly_list = list(y_train)
    
    # salvando os indexes dos dados anomalos
    anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == anomaly_class]
    
    # salvando os indexes dos dados normais
    #normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x != anomaly_class]
    normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x in train_class]
    
    # printando a quantidade de dados anomalos
    print('--- Anomaly_data_indexes ---')
    print(len(anomaly_data_indexes))
    
    # printando a quantidade de dados normais
    print('--- Normal_data_indexes ---')
    print(len(normal_data_indexes))
    
    # criando as variaveis para armazenar os dados e os labels dos dados normais e anormais
    X_normal_class, X_anomaly_class = [], []
    Y_normal_class, Y_anomaly_class = [], []
    
    # copiando os dados e os labels do conjunto de treinamento
    X_train = x_train.copy()
    Y_train = y_train.copy()
    
    '''    
    # retirando os dados anormais do conjunto de treinamento
    for i in range(len(X_train)):
        if(i in anomaly_data_indexes):
            X_anomaly_class.append(X_train[i])
            Y_anomaly_class.append(Y_train[i])
        else:
            X_normal_class.append(X_train[i])
            Y_normal_class.append(Y_train[i])
    '''     
    
    # juntando os dados normais e as anomalias
    for i in range(len(X_train)):
        if(i in anomaly_data_indexes):
            X_anomaly_class.append(X_train[i])
            Y_anomaly_class.append(Y_train[i])
        elif(i in normal_data_indexes):
            X_normal_class.append(X_train[i])
            Y_normal_class.append(Y_train[i])
    
    # transformando as variaveis em um array    
    X_normal = np.asarray(X_normal_class)
    Y_normal = np.asarray(Y_normal_class)
    
    # salvando os dados normais de treinamento nas variaveis finais
    X_train = X_normal.copy()
    Y_train = Y_normal.copy()
    
    # printando o formato das variaveis de saida
    print('X_train')
    print(X_train.shape)
    print('Y_train')
    print(Y_train.shape)
    
    
    # passa para o método os labels dos dados normais e recebe quem são os dados unicos e a quantidade de cada um
    unique, counts = np.unique(Y_train, return_counts=True)
    
    # criando um dicionario com os dados separados
    temp = dict(zip(unique, counts))
    
    # printando o dicionario
    print('counter_Y_normal')
    print(temp)
    
    
    ####################################################etapa dos dados de teste#############################################################
    
    # criando uma lista com os labels de teste
    anomaly_list = list(y_test)
    
    # salvando os indexes dos dados anomalos
    anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == anomaly_class]
    
    # salvando os indexes dos dados normais
    normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x in train_class]
    
    # printando a quantidade de dados anomalos
    print('--- Anomaly_data_indexes ---')
    print(len(anomaly_data_indexes))
    
    # printando a quantidade de dados normais
    print('--- Normal_data_indexes ---')
    print(len(normal_data_indexes))
    
    # criando as variaveis para armazenar os dados e os labels dos dados normais e anormais
    X_anomaly_class = []
    Y_anomaly_class = []
    
    # copiando os dados e os labels do conjunto de teste
    X_test = x_test.copy()
    Y_test = y_test.copy()
    
    # juntando os dados normais e as anomalias
    for i in range(len(X_test)):
        if((i in anomaly_data_indexes) or (i in normal_data_indexes)):
            X_anomaly_class.append(X_test[i])
            Y_anomaly_class.append(Y_test[i])
            
    # transformando as variaveis em um array    
    X_anomaly = np.asarray(X_anomaly_class)
    Y_anomaly = np.asarray(Y_anomaly_class)
    
    # copiando os dados e os labels do conjunto de teste
    X_test = X_anomaly.copy()
    Y_test = Y_anomaly.copy()
    
    # printando o formato dos labels das variaveis
    print('X_test')
    print(X_test.shape)
    print('Y_test')
    print(Y_test.shape)
    
    
    # passa para o método os labels dos dados normais e recebe quem são os dados unicos e a quantidade de cada um
    unique, counts = np.unique(Y_test, return_counts=True)
    
    # criando um dicionario com os dados separados
    temp = dict(zip(unique, counts))
    
    # printando o dicionario
    print('counter_Y_normal')
    print(temp)
    
    return X_train, Y_train, X_test, Y_test

def pre_processing_X_train_multiclass(train, target, anomaly_class):
    anomaly_list = list(target)
    anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == anomaly_class]
    normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x != anomaly_class]
    
    print('--- Anomaly_data_indexes ---')
#     print(anomaly_data_indexes)
    print(len(anomaly_data_indexes))
    
    print('--- Normal_data_indexes ---')
#     print(normal_data_indexes)
    print(len(normal_data_indexes))
    
    X_normal_class, X_anomaly_class = [], []
    Y_normal_class, Y_anomaly_class = [], []
    X_train = train.copy()
    Y_train = target.copy()
    for i in range(len(X_train)):
        if(i in anomaly_data_indexes):
#             print(i)
            X_anomaly_class.append(X_train[i])
            Y_anomaly_class.append(Y_train[i])
        else:
            X_normal_class.append(X_train[i])
            Y_normal_class.append(Y_train[i])
    X_normal = np.asarray(X_normal_class)
    X_anomaly = np.asarray(X_anomaly_class)
    
    Y_normal = np.asarray(Y_normal_class)
    Y_anomaly = np.asarray(Y_anomaly_class)
    
    print('x_normal')
    print(X_normal.shape)
    print('x_anomaly')
    print(X_anomaly.shape)
    
    print('y_normal')
    print(Y_normal.shape)
    print('y_anomaly')
    print(Y_anomaly.shape)
    
    return X_normal, Y_normal, X_anomaly, Y_anomaly    

def pre_processing_one_class_label(target, anomaly_class):    
    anomaly_list = list(target)
    anomaly_data_indexes = [i for i,x in enumerate(anomaly_list) if x == anomaly_class]
    normal_data_indexes = [i for i,x in enumerate(anomaly_list) if x != anomaly_class]
    
    print('--- Anomaly_data_indexes ---')
#     print(anomaly_data_indexes)
    print(len(anomaly_data_indexes))
    
    print('--- Normal_data_indexes ---')
#     print(normal_data_indexes)
    print(len(normal_data_indexes))
    
    one_class_labels = []
    for i in range(len(target)):
        if(i in anomaly_data_indexes):
            one_class_labels.append(constants.ANOMALY_DATA_REPRESENTATION)
        else:
            one_class_labels.append(constants.NORMAL_DATA_REPRESENTATION)

    one_class_target = np.asarray(one_class_labels)    
    return one_class_target
    

# def get_compiled_ypreds(y_test, y_preds, num_classes):
#     temp = np.ones((y_test.shape[0],), dtype=np.int)
#     temp = (-1 * temp)
#     nc_temp = num_classes - 1
#     for i in range(y_test.shape[0]):
#         for j in range(nc_temp):
#             if(y_preds[j][i] == NORMAL_DATA_REPRESENTATION):
#                 temp[i] = NORMAL_DATA_REPRESENTATION
#                 break
#     return temp

def get_compiled_ypreds_winner_takes_all(y_test, y_preds, num_classes):
    compiled_ypreds = np.zeros((y_test.shape[0],), dtype=np.int)
    nc_temp = num_classes - 1
    winner_takes_all = (constants.ANOMALY_CLASS * np.ones((y_test.shape[0],), dtype=np.int))
#     probs = np.zeros((y_test.shape[0],), dtype=np.int)
    for i in range(y_test.shape[0]):
        for j in range(nc_temp):
            if(y_preds[j][i] == constants.NORMAL_DATA_REPRESENTATION):
                compiled_ypreds[i] = constants.NORMAL_DATA_REPRESENTATION
                if(j > constants.ANOMALY_CLASS):
                    winner_takes_all[i] = j + 1
                else:
                    winner_takes_all[i] = j
    return compiled_ypreds, winner_takes_all

def get_compiled_ypreds(y_test, y_preds, num_classes):
    compiled_ypreds = np.zeros((y_test.shape[0],), dtype=np.int)
    nc_temp = num_classes - 1
    for i in range(y_test.shape[0]):
        for j in range(nc_temp):
            if(y_preds[j][i] == constants.NORMAL_DATA_REPRESENTATION):
                compiled_ypreds[i] = constants.NORMAL_DATA_REPRESENTATION
                break
    return compiled_ypreds 

def print_statistics(mcc, auc_roc, auc_prc, f_score):  
    ''' receive np.array of the values'''  
    print('\n------------------ STATISTICS -------------')
    print('MCC_MEAN')
    print(np.mean(mcc))
    print('MCC_SD')
    print(np.std(mcc))
    
    print('ROC_MEAN')
    print(np.mean(auc_roc))
    print('ROC_SD')
    print(np.std(auc_roc))
    
    print('PRC_MEAN')
    print(np.mean(auc_prc))
    print('PRC_SD')
    print(np.std(auc_prc))
    
    print('F_SCORE_MEAN')
    print(np.mean(f_score))
    print('F_SCORE_SD')
    print(np.std(f_score))

''' --- save images ---'''    
def save_img_init(X_test, decoded_imgs, name):
    n = 15  # how many digits we will display
    plt.figure(figsize=(20, 4))
#     predicted = y_pred_anomaly_data[:15,]
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name +'_anomaly_digit')

def save_img_init2(X_test, decoded_imgs, name):
    n = 15  # how many digits we will display
    plt.figure(figsize=(20, 4))
#     predicted = y_pred_anomaly_data[:15,]
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name +'_anomaly_digit')
    
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # display original
#         ax = plt.subplot(2, n, i + 1)
#         plt.imshow(X_test[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     
#         # display reconstruction
#         ax = plt.subplot(2, n, i + 1 + n)
#         plt.imshow(decoded_imgs[i].reshape(28, 28))
#         plt.text(0, 0, predicted[i], color='black', 
#                  bbox=dict(facecolor='white', alpha=1))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.savefig(name +'_anomaly_digit_detection')

def save_img_end_detection(X_test, decoded_imgs, name):
    n = 15  # how many digits we will display
    plt.figure(figsize=(20, 4))
    X_test_temp = X_test[-15:,]
    decoded_imgs_temp = decoded_imgs[-15:,]
#     predicted = y_pred_anomaly_data[-15:,]
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test_temp[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs_temp[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name +'_normal_digit')
    

def save_img_end_detection2(X_test, decoded_imgs, name):
    n = 15  # how many digits we will display
    plt.figure(figsize=(20, 4))
    X_test_temp = X_test[-15:,]
    decoded_imgs_temp = decoded_imgs[-15:,]
#     predicted = y_pred_anomaly_data[-15:,]
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test_temp[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs_temp[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name +'_normal_digit')
    
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # display original
#         ax = plt.subplot(2, n, i + 1)
#         plt.imshow(X_test_temp[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     
#         # display reconstruction
#         ax = plt.subplot(2, n, i + 1 + n)
#         plt.imshow(decoded_imgs_temp[i].reshape(28, 28))
#         plt.text(0, 0, predicted[i], color='black', 
#                  bbox=dict(facecolor='white', alpha=1))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.savefig(name +'_normal_digit_detection')

def save_img(X_test, decoded_imgs, name):
    save_img_init(X_test, decoded_imgs, name)
    save_img_end_detection(X_test, decoded_imgs, name)

def save_img_init_CIFAR(X_test, decoded_imgs, name):
    n = 15  # how many digits we will display
    plt.figure(figsize=(20, 4))
#     predicted = y_pred_anomaly_data[:15,]
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name +'_anomaly_digit')

def save_img_init_COIL(X_test, decoded_imgs, name):
    save_img_init2(X_test, decoded_imgs, name)
    save_img_end_detection2(X_test, decoded_imgs, name)
 
def save_img_end_detection_CIFAR(X_test, decoded_imgs, name):
    n = 15  # how many digits we will display
    plt.figure(figsize=(20, 4))
    X_test_temp = X_test[-15:,]
    decoded_imgs_temp = decoded_imgs[-15:,]
#     predicted = y_pred_anomaly_data[-15:,]
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test_temp[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs_temp[i].reshape(32, 32, 3))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name +'_normal_digit')
    
def save_img_CIFAR(X_test, decoded_imgs, name):
    save_img_init_CIFAR(X_test, decoded_imgs, name)
    save_img_end_detection_CIFAR(X_test, decoded_imgs, name)

''' separar os canais RGB'''
def get_channels(pixel):
    red = pixel[0]
    green = pixel[1]
    blue = pixel[2]
    return red, green, blue

def get_image_channels(input_image):
#     print('input_image')
#     print(input_image)
#     print(input_image.shape)
    
    red_image = np.zeros((input_image.shape[0], input_image.shape[1], 1))
    green_image = np.zeros((input_image.shape[0], input_image.shape[1], 1))
    blue_image = np.zeros((input_image.shape[0], input_image.shape[1], 1))
    
#     print('red_image')
#     print(red_image)
#     print(red_image.shape)
#      
#     print('green_image')
#     print(green_image)
#     print(green_image.shape)
#      
#     print('blue_image')
#     print(blue_image)
#     print(blue_image.shape)
    
    # get row number
    for rownum in range(len(input_image)):
        for colnum in range(len(input_image[rownum])):
            red, green, blue = get_channels(input_image[rownum][colnum])
            red_image[rownum][colnum] = red
            green_image[rownum][colnum] = green
            blue_image[rownum][colnum] = blue
    
#     print('red_image')
#     print(red_image)
#     print(red_image.shape)
#      
#     print('green_image')
#     print(green_image)
#     print(green_image.shape)
#      
#     print('blue_image')
#     print(blue_image)
#     print(blue_image.shape)
    
#     teste = np.concatenate((red_image, green_image, blue_image), axis=2)
#     
#     print('teste')
#     print(teste)
#     print(teste.shape)
    
    
    return red_image, green_image, blue_image

def average_grayscale(pixel):
    return (pixel[0] + pixel[1] + pixel[2]) / 3
#     return np.average(pixel)

def weighted_average_grayscale(pixel):
    return (0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2])

def grayscale_average_conversion(input_image):
    grey_image = np.zeros((input_image.shape[0], input_image.shape[1], 1)) # init 2D numpy array
    # get row number
    for rownum in range(len(input_image)):
        for colnum in range(len(input_image[rownum])):
            grey_image[rownum][colnum] = average_grayscale(input_image[rownum][colnum])
    return grey_image

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def grayscale_weighted_average_conversion(input_image):
    grey_image = np.zeros((input_image.shape[0], input_image.shape[1], 1)) # init 2D numpy array
    # get row number
    for rownum in range(len(input_image)):
        for colnum in range(len(input_image[rownum])):
            grey_image[rownum][colnum] = weighted_average_grayscale(input_image[rownum][colnum])
    return grey_image

def pre_processing_subtract_mean(X_train, X_test):
    
#     print('X_train')
#     print(X_train.shape)
#     print(X_train[0].shape)
#     print(X_train[0])
    
    images_train = X_train.shape[0]
    images_test = X_test.shape[0]
    rows = X_train.shape[1]
    cols = X_train.shape[2]
    channels = X_train.shape[3]
#     print('aqui')
    
    mean = np.zeros((rows, cols, channels))
    
    for imagenum in range(images_train):
        for rownum in range(rows):
            for colnum in range(cols):
                for channel in range(channels):
                    mean[rownum][colnum][channel] = mean[rownum][colnum][channel] + X_train[imagenum][rownum][colnum][channel]
                    
    mean = mean / float(images_train)
    
#     print('mean')
#     print(mean.shape)
#     print(mean)
    
    X_train_temp = np.zeros((images_train, rows, cols, channels))
    X_test_temp = np.zeros((images_test, rows, cols, channels))
    
#     print('X_temp')
#     print(X_temp.shape)
    
#     print('X_train_temp')
    for i in range(images_train):
        X_train_temp[i] = np.subtract(X_train[i], mean)
    
#     print('X_test_temp')    
    for i in range(images_test):
        X_test_temp[i] = np.subtract(X_test[i], mean)
    
#     print('X_train_temp')
#     print(X_train_temp.shape)
#     print(X_train_temp[0].shape)
#     print(X_train_temp[0])
#     print('min')
#     print(np.min(X_train_temp))
#     print('max')
#     print(np.max(X_train_temp))
#     
#     print('X_test_temp')
#     print(X_test_temp.shape)
#     print(X_test_temp[0].shape)
#     print(X_test_temp[0])
#     print('min')
#     print(np.min(X_test_temp))
#     print('max')
#     print(np.max(X_test_temp))
    
#     print('\nNormalize')
    X_train_norm = np.zeros((images_train, rows, cols, channels))
    X_test_norm = np.zeros((images_test, rows, cols, channels))
    
#     print('X_train_temp')
    min_train = np.min(X_train_temp)
    max_train = np.max(X_train_temp)
    for i in range(images_train):
        X_train_norm[i] = ((X_train_temp[i] - min_train)/(max_train - min_train))
    
#     print('X_test_temp')    
    min_test = np.min(X_test_temp)
    max_test = np.max(X_test_temp)
    for i in range(images_test):
        X_test_norm[i] = ((X_test_temp[i] - min_test)/(max_test - min_test))
    
#     print('\nDEPOIS')
#     print('X_train_temp')
#     print(X_train_norm.shape)
#     print(X_train_norm[0].shape)
#     print(X_train_norm[0])
#     print('min')
#     print(np.min(X_train_norm))
#     print('max')
#     print(np.max(X_train_norm))
#     
#     print('X_test_temp')
#     print(X_test_norm.shape)
#     print(X_test_norm[0].shape)
#     print(X_test_norm[0])
#     print('min')
#     print(np.min(X_test_norm))
#     print('max')
#     print(np.max(X_test_norm))
    
    return X_train_norm, X_test_norm
    
    