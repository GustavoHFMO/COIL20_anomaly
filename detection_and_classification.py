#!/usr/local/bin/python

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Flatten
from keras.models import Model, Sequential
from keras.datasets import mnist
import numpy as np
import keras

from keras.models import load_model
from keras import optimizers
from keras import losses

from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV

import util, models, constants
from sklearn.metrics.classification import accuracy_score

import operator
from sklearn.cross_validation import train_test_split


def binary_cnns_anomaly_detection(X_train, Y_train, X_test, Y_test, dataset_name, num_classes):
    y_preds = []
#     one_indexes = []
    for i in range(num_classes):
        if(i == constants.ANOMALY_CLASS):
            continue
  
        print('INDEX: ' + str(i))
        one_class_y_train = util.pre_processing_one_class_label(Y_train, i)
         
        print('X_train')
        print(X_train.shape)
        print('y_train')
        print(Y_train.shape)
        print('one_class_y_train')
        print(one_class_y_train.shape)
         
        unique, counts = np.unique(one_class_y_train, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_one_class_y_train')
        print(temp)
        
        if(dataset_name == 'MNIST'):
            name_model = 'MNIST_binary_CNN_' + str(constants.OPT_FUNC) +'_without_' + str(i) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        elif(dataset_name == 'CIFAR'):
            name_model = 'CIFAR_binary_CNN_' + str(constants.OPT_FUNC) +'_without_' + str(i) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        elif(dataset_name == 'CIFAR100'):
            name_model = 'CIFAR100_binary_CNN_' + str(constants.OPT_FUNC) +'_without_' + str(i) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        print('\nModel: ' + name_model)

        ''' option: use validation in training phase'''
#             X_train_temp, x_valid, y_train_temp, y_valid = train_test_split(X_train, one_class_y_train, test_size=0.20, random_state=constants.RANDOM_STATE)
        try:
            print('Load..')    
            clf = load_model(name_model)
        except:
            print('TRAINING..')
            clf = models.cnn_binary_clf(X_train, one_class_y_train, dataset_name)
#                 clf = models.cnn_binary_clf_validation(X_train_temp, y_train_temp, x_valid, y_valid, dataset_name)
            clf.save(name_model)
        
        # predict
        print('PRED..')
        y_pred = clf.predict(X_test)
        
        unique, counts = np.unique(y_pred, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_y_pred')
        print(temp)
         
#         ones_temp = list(y_pred)
#         temp = [i for i,x in enumerate(ones_temp) if x == constants.NORMAL_DATA_REPRESENTATION]
#         one_indexes.append(temp)
     
        ''' appending preds'''
        y_preds.append(y_pred)
         
    ''' compiling ypreds '''
    winner_takes_all = None    
    compiled_ypreds, winner_takes_all = util.get_compiled_ypreds_winner_takes_all(Y_test, y_preds, num_classes)
     
    unique, counts = np.unique(compiled_ypreds, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_compiled_ypreds')
    print(temp)
    
    ''' one_class representation of the Y_test''' 
    final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)
     
    unique, counts = np.unique(final_one_class_y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_final')
    print(temp)
     
    print('compiled_ypreds')
    print(compiled_ypreds)
    temp = np.asarray(compiled_ypreds)
    print(temp.shape)
 
    print('\nPrint informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, compiled_ypreds)

    print('\n------------------ Winner takes all: ------------------')
    print(winner_takes_all)
      
    unique, counts = np.unique(winner_takes_all, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_winner_takes_all')
    print(temp)
      
    unique, counts = np.unique(Y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_y_test')
    print(temp)
      
    print('ACCURACY - winner_takes_all')
    print(accuracy_score(Y_test, winner_takes_all))
    
    # compiled_ypreds: vector with the anomaly detection compiled from the binary_clfs 
    # anomaly vector representation: formed with ANOMALY_DATA_REPRESENTATION and NORMAL_DATA_REPRESENTATION
#     return compiled_ypreds
    ''' counting if more than one binary_cnn classifies an instance as normal''' 
#     size = 0
#     for i in range(len(one_indexes)):
#         classe = (i + 1)
#         print('Class: ' + str(classe))
#         print('len: ')
#         print(len(one_indexes[i]))
#         size = size + len(one_indexes[i])
      
#     print('final_size')
#     print(size)   
#           
#     print('------------------ FINAL -----------------')
#     final = []
#     a1 = one_indexes[0]
#     a2 = one_indexes[1]
#     a3 = one_indexes[2]
#     a4 = one_indexes[3]
#     a5 = one_indexes[4]
#     a6 = one_indexes[5]
#     a7 = one_indexes[6]
#     a8 = one_indexes[7]
#     a9 = one_indexes[8]
#     final = a1 + a2 + a3+ a4+ a5+ a6+ a7+ a8+ a9
#       
#     unique, counts = np.unique(final, return_counts=True)
#     temp = dict(zip(unique, counts))
#     print('counter_final')
#     print(temp)
    
    return compiled_ypreds

def multi_ocsvm_anomaly_detection(X_train, Y_train, X_test, Y_test, clf_type, dataset_name, num_classes):
    y_preds = []
    for i in range(num_classes):
        if(i == constants.ANOMALY_CLASS):
            continue
  
        print('INDEX: ' + str(i))
        one_class_y_train = util.pre_processing_one_class_label(Y_train, i)
         
        print('X_train')
        print(X_train.shape)
        print('y_train')
        print(Y_train.shape)
        print('one_class_y_train')
        print(one_class_y_train.shape)
         
        unique, counts = np.unique(one_class_y_train, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_one_class_y_train')
        print(temp)
                
        X_normal, Y_normal, _, _ = util.pre_processing_X_train_multiclass(X_train, Y_train, i)
        
        print('X_normal')
        print(X_normal.shape)
        print('Y_normal')
        print(Y_normal.shape)
        print('one_class_y_train')
        print(one_class_y_train.shape)
         
        unique, counts = np.unique(Y_normal, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_Y_normal')
        print(temp)

        if(clf_type == 'DEF_OCSVM'):
            if(dataset_name == 'MNIST'):
                name_model = 'MNIST_def_ocsvm_without_'+ str(i) +'_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
            elif(dataset_name == 'CIFAR'):
                name_model = 'CIFAR_def_ocsvm_without_'+ str(i) +'_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
            elif(dataset_name == 'CIFAR100'):
                name_model = 'CIFAR100_def_ocsvm_without_'+ str(i) +'_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
            elif(dataset_name == 'SI'):
                name_model = 'SI_def_ocsvm_without_'+ str(i) +'_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
            print('\nModel: ' + name_model)
#             try:
#                 print('LOAD...')
#                 clf = joblib.load(name_model)
#             except:    
            print('TRAINING...')
            clf = svm.OneClassSVM()
            clf = clf.fit(X_normal) 
                 
#                 print('SAVE...')
#                 joblib.dump(clf, name_model)
        elif(clf_type == 'GRID_OCSVM'):
            if(dataset_name == 'MNIST'):
                name_model = 'MNIST_grid_ocsvm_without_'+ str(i) +'_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
            elif(dataset_name == 'CIFAR'):
                name_model = 'CIFAR_grid_ocsvm_without_'+ str(i) +'_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
            print('\nModel: ' + name_model)

            try:
                print('LOAD...')
                clf = joblib.load(name_model)
            except:
        #         param_grid = {'nu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, gamma_auto], }
        #         clf = GridSearchCV(OneClassSVM(kernel='rbf'), param_grid, scoring='f1')
                 
                print('TRAINING...')
                grid_clf = GridSearchCV(OneClassSVM(kernel='rbf'), cv=3, n_jobs=-1, param_grid = {'nu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}, scoring='f1')
                grid_clf = grid_clf.fit(X_normal)
                 
                clf = grid_clf.best_estimator_
                print('BEST')
                print(clf)
                 
                print('SAVE...')
                joblib.dump(clf, name_model)
 
        # predict
        print('PRED..')
        y_pred = clf.predict(X_test)
        # transforming to anomaly data 0 representation
        for i in range(len(y_pred)):
            if(y_pred[i] == -1):
                y_pred[i] = constants.ANOMALY_DATA_REPRESENTATION
        
        unique, counts = np.unique(y_pred, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_y_pred')
        print(temp)
     
        ''' appending preds'''
        y_preds.append(y_pred)
         
    ''' compiling ypreds '''
    compiled_ypreds = util.get_compiled_ypreds(Y_test, y_preds, num_classes)
     
    unique, counts = np.unique(compiled_ypreds, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_compiled_ypreds')
    print(temp)
    
    ''' one_class representation of the Y_test''' 
    final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)
     
    unique, counts = np.unique(final_one_class_y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_final')
    print(temp)
     
    print('compiled_ypreds')
    print(compiled_ypreds)
    temp = np.asarray(compiled_ypreds)
    print(temp.shape)
 
    print('\nPrint informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, compiled_ypreds)
    
    return compiled_ypreds

def multiclass_cnn_classifier(X_train, Y_train, X_test, Y_test, dataset_name, num_classes, batch_normalization, dropout):
    print('\n------------------ Multiclass classifier: ------------------')
     
    print('Y_TRAIN')
    print(Y_train)
     
    unique, counts = np.unique(Y_train, return_counts=True)
    temp = dict(zip(unique, counts))
    print('Y_train_classes_antes')
    print(temp)  
     
    print('Y_train_shape_antes')
    print(Y_train.shape)
     
    # convert class vectors to binary class matrices
    y_train_matrix = keras.utils.to_categorical(Y_train, num_classes)
    y_test_matrix = keras.utils.to_categorical(Y_test, num_classes)
     
    print('Y_train_shape_depois')
    print(y_train_matrix.shape)
    print(y_train_matrix)
     
    ''' training multiclass cnn'''
    if((dropout) and (batch_normalization)):
        if(dataset_name == 'MNIST'):
            name_model = 'MNIST_multiclass_CNN_BN_Dropout_' + str(constants.OPT_FUNC) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH_MULTI_CLASS) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        elif(dataset_name == 'CIFAR'):
            name_model = 'CIFAR_multiclass_CNN_BN_Dropout_' + str(constants.OPT_FUNC) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH_MULTI_CLASS) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        print('\nModel: ' + name_model)
            
        try:
            print('Load..')    
            clf = load_model(name_model)
        except:
            print('TRAINING..')
            clf = models.multiclass_cnn_bn_dropout(X_train, y_train_matrix, dataset_name, num_classes)
            clf.save(name_model)
    elif((not dropout) and (batch_normalization)):
        if(dataset_name == 'MNIST'):
            name_model = 'MNIST_multiclass_CNN_BN_' + str(constants.OPT_FUNC) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH_MULTI_CLASS) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        elif(dataset_name == 'CIFAR'):
            name_model = 'CIFAR_multiclass_CNN_BN_' + str(constants.OPT_FUNC) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH_MULTI_CLASS) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        print('\nModel: ' + name_model)
            
        try:
            print('Load..')    
            clf = load_model(name_model)
        except:
            print('TRAINING..')
            clf = models.multiclass_cnn_bn(X_train, y_train_matrix, dataset_name, num_classes)
            clf.save(name_model)
    elif((dropout) and (not batch_normalization)):
        if(dataset_name == 'MNIST'):
            name_model = 'MNIST_multiclass_CNN_Dropout_' + str(constants.OPT_FUNC) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH_MULTI_CLASS) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        elif(dataset_name == 'CIFAR'):
            name_model = 'CIFAR_multiclass_CNN_Dropout_' + str(constants.OPT_FUNC) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH_MULTI_CLASS) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        print('\nModel: ' + name_model)
            
        try:
            print('Load..')    
            clf = load_model(name_model)
        except:
            print('TRAINING..')
            clf = models.multiclass_cnn_dropout(X_train, y_train_matrix, dataset_name, num_classes)
            clf.save(name_model)
    else: # without dropout and batch_normalization
        if(dataset_name == 'MNIST'):
            name_model = 'MNIST_multiclass_CNN_' + str(constants.OPT_FUNC) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH_MULTI_CLASS) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        elif(dataset_name == 'CIFAR'):
            name_model = 'CIFAR_multiclass_CNN_' + str(constants.OPT_FUNC) +'_filter' + str(constants.NB_FILTERS) + '_epoch'+ str(constants.NB_EPOCH_MULTI_CLASS) + '_anomaly_' + str(constants.ANOMALY_CLASS) +'.h5'
        print('\nModel: ' + name_model)
            
        try:
            print('Load..')    
            clf = load_model(name_model)
        except:
            print('TRAINING..')
            clf = models.multiclass_cnn(X_train, y_train_matrix, dataset_name, num_classes)
            clf.save(name_model)
 
    # predict
    print('PRED..')
    y_pred = clf.predict(X_test)
     
    ''' tranforming results from matrix to array with the max prob class: max_indexes are the classes'''
    max_indexes, max_values = [], []
    for i in range(len(y_pred)):
        max_index, max_value = max(enumerate(y_pred[i]), key=operator.itemgetter(1))
        max_indexes.append(max_index)
        max_values.append(max_value)
     
    unique, counts = np.unique(max_indexes, return_counts=True)
    temp = dict(zip(unique, counts))
    print('Multiclass_y_pred')
    print(temp)    
     
    unique, counts = np.unique(Y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('y_test')
    print(temp)
      
    print('ACCURACY - multiclass_cnn')
    print(accuracy_score(Y_test, max_indexes))
    
    score = clf.evaluate(X_test, y_test_matrix, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    ''' 
    max_indexes: index of the class with max softmax prob
    max_values: max softmax prob
    '''
    return max_indexes, max_values

def multiclass_cnn_anomaly_detection(X_train, Y_train, X_test, Y_test, dataset_name, num_classes, batch_normalization, dropout):   
    max_indexes, max_values = multiclass_cnn_classifier(X_train, Y_train, X_test, Y_test, dataset_name, num_classes, batch_normalization, dropout)
    
    print('\nRESULTS Multi_class_anomaly_detection: ')
    limits = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for limit in limits:
        print('\nThreshold: ' + str(limit))
        cnn_threshold_pred = util.anomaly_detection_cnn_threshold(max_indexes, max_values, limit)
        acc = accuracy_score(Y_test, cnn_threshold_pred)
        ''' one_class representation of the Y_test''' 
        final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)        
        one_class_y_pred = util.pre_processing_one_class_label(cnn_threshold_pred, constants.ANOMALY_CLASS)
        
        print('\nPrint informations:')
        print('accuracy: ' + str(acc))
        ''' both arrays must be in one_class form '''
        mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, one_class_y_pred)
    
    return one_class_y_pred

def multiclass_classification(X_train, Y_train, X_test, Y_test, compiled_ypreds, dataset_name, num_classes):
    max_indexes, max_values = multiclass_cnn_classifier(X_train, Y_train, X_test, Y_test, dataset_name, num_classes)
     
    print('len - normal/anomalias')
    print(len(compiled_ypreds))
    print('len - multi_clf')
    print(len(max_indexes))
      
    ''' 
    compiling multi_class_y_pred(max_indexes) and anomaly_pred(compiled_ypreds)
    if anomaly detector classifies as normal data: uses the class of the mult_clf
    else: classifies as an anomaly  
    '''
    multiclass_with_anomaly_detection = []
    for i in range(len(max_indexes)):
        if(compiled_ypreds[i] == constants.NORMAL_DATA_REPRESENTATION):
            multiclass_with_anomaly_detection.append(max_indexes[i])
        else:
            multiclass_with_anomaly_detection.append(constants.ANOMALY_CLASS)
#             multi_class_y_pred.append(compiled_ypreds[i])
    
    unique, counts = np.unique(multiclass_with_anomaly_detection, return_counts=True)
    temp = dict(zip(unique, counts))
    print('multiclass_with_anomaly_detection')
    print(temp)        
      
    print('\nACCURACY - multiclass_with_anomaly_detection')
    print(accuracy_score(Y_test, multiclass_with_anomaly_detection))
    
#     score = clf.evaluate(X_test, y_test_matrix, verbose=0)
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])
    
    print('\nRESULTS multiclass_with_anomaly_detection: ')
#     cnn_threshold_pred = util.anomaly_detection_cnn_threshold(max_indexes, max_values, 0.9)
#     acc = accuracy_score(Y_test, cnn_threshold_pred)
#     ''' one_class representation of the Y_test''' 
    final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)        
    one_class_y_pred = util.pre_processing_one_class_label(multiclass_with_anomaly_detection, constants.ANOMALY_CLASS)
    
    print('\nPrint informations:')
#     print('accuracy: ' + str(acc))
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, one_class_y_pred)

def CAE_anomaly_detection(X_train, Y_train, X_test, Y_test, dataset_name, num_classes, execution):
    '''
    :obj: Faz o procedimento completo de detecção de anomalia usando CAE
    :param X_train: Dados para treinamento 
    :param Y_train: Labels para treinamento
    :param X_test: Dados para teste
    :param Y_test: Labels do teste
    :param dataset_name: Nome do dataset que será executado
    :param num_classes: Numero de classes
    :return: y_pred_anomaly_data_quartile: retorna o erro de reconstrução da anomalia
    '''
    
    # Criando um nome para saber o que ta sendo executado
    if(dataset_name == 'MNIST'):
        name = 'CAE_MNIST_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_model' + str(execution)
    elif(dataset_name == 'CIFAR'):
        name = 'CAE_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_model' + str(execution)
    elif(dataset_name == 'CIFAR_gray'):
        name = 'CAE_CIFAR_gray_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_model' + str(execution)
    elif(dataset_name == 'CIFAR100'):
        name = 'CAE_CIFAR100_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_model' + str(execution)
    elif(dataset_name == 'SVHN'):
        name = 'CAE_SVHN_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_model' + str(execution)
    elif(dataset_name == 'SI'):
        name = 'CAE_SI_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_model' + str(execution)
    elif(dataset_name == 'SI_invert'):
        name = 'CAE_SI_invert_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_model' + str(execution)
    elif(dataset_name == 'COIL20'):
        name = 'COIL20_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_model' + str(execution)
    name_model = name +'.h5'
    print('\nModel: ' + name_model)


    # Carregando um modelo para treinar os dados
    try:
        print('Load..')    
        autoencoder = load_model(name_model)
    except:
        print('TRAINING..')
#         autoencoder, encoded, decoded, input_img = models.cae_flatten(X_train, dataset_name)
#         autoencoder, encoded, decoded, input_img = models.convdeconv_cae_flatten(X_train, dataset_name)
        #autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool(X_train, dataset_name)
        ''' layers_valid '''
#         autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool_layers_valid(X_train, dataset_name)
        
        ''' layers '''
        autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool_layers(X_train, dataset_name)
#         autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool_layers_zca(X_train, dataset_name)
        
        ''' sparse '''
#         autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool_layers_sparse(X_train, dataset_name)
        ''' unet'''
#         autoencoder, encoded, decoded, input_img = models.get_unet(X_train, dataset_name)
        
        ''' grayscale'''
#         autoencoder, encoded, decoded, input_img = models.cae_flatten_grayscale(X_train, dataset_name)
        
#         autoencoder, encoded, decoded, input_img = models.cae_flatten_deep10(X_train, dataset_name)
        
        ''' antigo '''
        '''
        if(constants.DEEP == 1):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep1(X_train, dataset_name)
        elif(constants.DEEP == 3):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep3(X_train, dataset_name)
        elif(constants.DEEP == 5):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep5(X_train, dataset_name)
        elif(constants.DEEP == 10):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep10(X_train, dataset_name)
        '''
        autoencoder.save(name_model)
    
    # com o modelo treinado computando a reconstrucao para o conjunto de treinamento
    print('PRED_TRAIN..')
    decoded_train = autoencoder.predict(X_train)
#     name_train = name + '_train'
#     util.save_img_CIFAR(X_train, decoded_train, name_train)

    # reconstruindo os dados de teste
    print('PRED..')
    decoded_imgs = autoencoder.predict(X_test)    
#     name_test = name + '_test'
#     util.save_img_CIFAR(X_test, decoded_imgs, name_test)
    if((dataset_name == 'CIFAR') or (dataset_name == 'CIFAR100') or (dataset_name == 'SVHN')):
        util.save_img_CIFAR(X_test, decoded_imgs, dataset_name)
    elif((dataset_name == 'MNIST') or (dataset_name == 'SI') or (dataset_name == 'SI_invert')):
        util.save_img(X_test, decoded_imgs, dataset_name)
    elif((dataset_name == 'COIL20') or (dataset_name == 'CIFAR_gray')):
        util.save_img_init_COIL(X_test, decoded_imgs, dataset_name)
    
    # computando o erro de reconstrução para o conjunto de treinamento
    print('GRIDSEARCH..')
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile, error_train = util.gridsearch_error_logloss_debug(X_train, decoded_train)
    ''' best_alpha'''
    alpha_mean = er_mean
    alpha_quartile = third_quartile
    
    
    # Computando o erro do conjunto de teste para as anomalias 
    '''anomaly detection'''
    print('\nMean: ')
    y_pred_anomaly_data_mean = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_mean)
    
    # terceiro quartil do erro de reconstrução da anomalia
    print('\nThird quartile: ')
    y_pred_anomaly_data_quartile = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_quartile)
    
    ''' one_class representation of the Y_test''' 
    final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)      
    
    ''' new prints for test'''
    print('final_one_class_y_test')
    print(final_one_class_y_test)
    print('shape[0]')
    print(final_one_class_y_test.shape[0])
    
    error_test_normal = []
    error_test_anomaly = []
    
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile, error = util.gridsearch_error_logloss_debug(X_test, decoded_imgs)
    
    # erros de reconstrução para o conjunto de teste
    print('\nTEST')
    print('er_mean')
    print(er_mean)
    print('er_min')
    print(er_min)
    print('er_max')
    print(er_max)
    print('first_quartile')
    print(first_quartile)
    print('second_quartile')
    print(second_quartile)
    print('third_quartile')
    print(third_quartile)
#     print('\nERROR')
#     print(error)
    
    #agrupando o conjunto de erros para o tipo de dados - normal ou anormal
    for i in range(final_one_class_y_test.shape[0]): 

        if(final_one_class_y_test[i] == constants.ANOMALY_DATA_REPRESENTATION):
            error_test_anomaly.append(error[i])
        else:
            error_test_normal.append(error[i])
    
    
    # erro de teste da anomalia
    print('\nerror_test_anomaly')
    print(error_test_anomaly)
    print(len(error_test_anomaly))
    
    # variaveis que contem o erro de teste da anomalia
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.calc_error_metrics(error_test_anomaly)
    
    
    # erro de teste para dados normais
    print('\nerror_test_normal')
    print(error_test_normal) 
    print(len(error_test_normal))   
    
    # variaveis que contem o erro de teste para dados normais
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.calc_error_metrics(error_test_normal) 
    ''' end of new prints'''  
    
    
    # printando métricas para media do erro de reconstruçao da anomalia
    print('\nPrint informations - Mean informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_mean)
    
    
    # printando métricas para o segundo quartil do erro de reconstruçao da anomalia
    print('\nPrint informations - Third quartile informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_quartile)
    
    
    ################################################### plot do erro de reconstrução ##########################################################
    import matplotlib.pyplot as plt
    import numpy as np
    import pylab as P
    
    plt.figure(figsize =(20,15))
    
    qtd_normal = len(error_test_normal)
    qtd_anomaly = len(error_test_anomaly)
    
    plt.scatter(range(qtd_normal), error_test_normal, color = 'blue', label = str(qtd_normal) + ' normal data')
    plt.scatter(range(qtd_anomaly), error_test_anomaly, color = 'red', label = str(qtd_anomaly) + ' anomaly data')
    
    mean_normal = np.mean(error_test_normal)
    mean_anomaly = np.mean(error_test_anomaly)
    
    plt.axhline(y=mean_normal, linewidth=2, color = 'blue', label = 'normal mean: ' + str(mean_normal))
    plt.axhline(y=mean_anomaly, linewidth=2, color = 'red', label = 'anomaly mean: ' + str(mean_anomaly))
    
    plt.xlabel('instance number')
    plt.ylabel('logloss')
    
    plt.title(dataset_name + ' - reconstruction error')
    
    texto = (" mcc: %.2f | auc_roc:  %.2f | auc_prc: %.2f | f_score: %.2f  " %(mcc, auc_roc, auc_prc, f_score))
    plt.annotate(texto, xy=(0.45, 0.85), xytext=(0, 0), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', size=14, ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))
    
    plt.style.use('seaborn-poster')
    plt.legend()
    
    plt.savefig(dataset_name + '_err_recons.png')
    #plt.show()
    ########################################################################################################################################
    
    
    ################################################### plot do erro de reconstrução ordenado ##############################################
    
    plt.figure(figsize =(20,15))
    
    sort_error_test_normal = np.sort(error_test_normal)
    sort_error_test_anomaly = np.sort(error_test_anomaly)
    
    qtd_normal = len(error_test_normal)
    qtd_anomaly = len(error_test_anomaly)
    
    plt.scatter(range(qtd_normal), sort_error_test_normal, color = 'blue', label = str(qtd_normal) + ' normal data')
    plt.scatter(range(qtd_anomaly), sort_error_test_anomaly, color = 'red', label = str(qtd_anomaly) + ' anomaly data')
    
    plt.xlabel('instance number')
    plt.ylabel('logloss')
    
    plt.title(dataset_name + ' - order reconstruction error')
    
    texto = (" mcc: %.2f | auc_roc:  %.2f | auc_prc: %.2f | f_score: %.2f  " %(mcc, auc_roc, auc_prc, f_score))
    plt.annotate(texto, xy=(0.45, 0.85), xytext=(0, 0), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', size=14, ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))
    
    plt.style.use('seaborn-poster')
    plt.legend()
    
    plt.savefig(dataset_name + '_sort_err_recons.png')
    #plt.show()
    ########################################################################################################################################
    
    
    ################################################### plot do histograma #################################################################
    import math
    plt.figure(figsize =(20,15))
    
    error_test_anomaly2 = []
    for i in range(len(error_test_anomaly)):
        if(math.isnan(error_test_anomaly[i]) or math.isinf(error_test_anomaly[i])):
            continue
        else:
            error_test_anomaly2.append(error_test_anomaly[i])
        
    # anomaly
    mu_anomaly = np.mean(error_test_anomaly2)
    sigma_anomaly = np.std(error_test_anomaly2)
    n, bins, patches = P.hist(error_test_anomaly2, 50, normed=1, histtype='stepfilled', label = 'anomaly', alpha = 0.5)
    #y = P.normpdf(bins, mu_anomaly, sigma_anomaly)
    #l = P.plot(bins, y, linewidth=1.5, label = 'anomaly')
    n1 = np.max(n)
    
    
    error_test_normal2 = []
    for i in range(len(error_test_normal)):
        if(math.isnan(error_test_normal[i]) or math.isinf(error_test_normal[i])):
            continue
        else:
            error_test_normal2.append(error_test_normal[i])
    
    # normal
    mu_normal = np.mean(error_test_normal2)
    sigma_normal = np.std(error_test_normal2)
    n, bins, patches = P.hist(error_test_normal2, 50, normed=1, histtype='stepfilled', label = 'normal', alpha = 0.5)
    #y = P.normpdf( bins, mu_normal, sigma_normal)
    #l = P.plot(bins, y, linewidth=1.5, label = 'normal')
    
    n2 = np.max(n)
    
    max = np.max([n1, n2]) + 0.1
    P.ylim([0, max])
    #P.xlim([0,1000])
    
    plt.title(dataset_name + ' - histogram')
    plt.legend()
    
    plt.savefig(dataset_name + '_histogram.png')
    #plt.show()
    ########################################################################################################################################
 
    return y_pred_anomaly_data_quartile

def CAE_anomaly_detection2(X_train, Y_train, X_test, Y_test, dataset_name, num_classes):
    '''
    :obj: Faz o procedimento completo de detecção de anomalia usando CAE
    :param X_train: Dados para treinamento 
    :param Y_train: Labels para treinamento
    :param X_test: Dados para teste
    :param Y_test: Labels do teste
    :param dataset_name: Nome do dataset que será executado
    :param num_classes: Numero de classes
    :return: y_pred_anomaly_data_quartile: retorna o erro de reconstrução da anomalia
    '''
    
    # Criando um nome para saber o que ta sendo executado
    if(dataset_name == 'MNIST'):
        name = 'CAE_MNIST_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS)
    elif(dataset_name == 'CIFAR'):
        name = 'CAE_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS)
    elif(dataset_name == 'CIFAR100'):
        name = 'CAE_CIFAR100_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS)
    elif(dataset_name == 'SVHN'):
        name = 'CAE_SVHN_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS)
    name_model = name +'.h5'
    print('\nModel: ' + name_model)


    # treinando ou carregando um modelo treinado
    try:
        print('Load..')    
        autoencoder = load_model(name_model)
    except:
        print('TRAINING..')
#         autoencoder, encoded, decoded, input_img = models.cae_flatten(X_train, dataset_name)
#         autoencoder, encoded, decoded, input_img = models.convdeconv_cae_flatten(X_train, dataset_name)
        autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool(X_train, dataset_name)
        ''' layers_valid '''
#         autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool_layers_valid(X_train, dataset_name)
        
        ''' layers '''
#         autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool_layers(X_train, dataset_name)
#         autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool_layers_zca(X_train, dataset_name)
        
        ''' sparse '''
#         autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_no_pool_layers_sparse(X_train, dataset_name)
        ''' unet'''
#         autoencoder, encoded, decoded, input_img = models.get_unet(X_train, dataset_name)
        
        ''' grayscale'''
#         autoencoder, encoded, decoded, input_img = models.cae_flatten_grayscale(X_train, dataset_name)
        
#         autoencoder, encoded, decoded, input_img = models.cae_flatten_deep10(X_train, dataset_name)
        
        ''' antigo '''
        if(constants.DEEP == 1):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep1(X_train, dataset_name)
        elif(constants.DEEP == 3):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep3(X_train, dataset_name)
        elif(constants.DEEP == 5):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep5(X_train, dataset_name)
        elif(constants.DEEP == 10):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep10(X_train, dataset_name)
        autoencoder.save(name_model)
    
    # predizendo todo o conjunto de treinamento
    print('PRED_TRAIN..')
    decoded_train = autoencoder.predict(X_train)



    ###################################################################################################################################
    
    # USAR O SVM AQUI
    clf = svm.SVC(kernel='linear', C = 1.0, probability=True)
    clf.fit(decoded_train, Y_train)
    
    ###################################################################################################################################



    # predizendo todo o conjunto de teste
    print('PRED..')
    decoded_imgs = autoencoder.predict(X_test)    
    
    # computando o erro de reconstrução para o conjunto de treinamento
    print('GRIDSEARCH..')
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile, error_train = util.gridsearch_error_logloss_debug(X_train, decoded_train)
    ''' best_alpha'''
    alpha_mean = er_mean
    alpha_quartile = third_quartile
    
    
    # Computando o erro do conjunto de teste para as anomalias 
    '''anomaly detection'''
    print('\nMean: ')
    y_pred_anomaly_data_mean = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_mean)
    
    # terceiro quartil do erro de reconstrução da anomalia
    print('\nThird quartile: ')
    y_pred_anomaly_data_quartile = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_quartile)
    
    ''' one_class representation of the Y_test''' 
    final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)      
    
    ''' new prints for test'''
    print('final_one_class_y_test')
    print(final_one_class_y_test)
    print('shape[0]')
    print(final_one_class_y_test.shape[0])
    
    error_test_normal = []
    error_test_anomaly = []
    
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile, error = util.gridsearch_error_logloss_debug(X_test, decoded_imgs)
    
    # erros de reconstrução para o conjunto de teste
    print('\nTEST')
    print('er_mean')
    print(er_mean)
    print('er_min')
    print(er_min)
    print('er_max')
    print(er_max)
    print('first_quartile')
    print(first_quartile)
    print('second_quartile')
    print(second_quartile)
    print('third_quartile')
    print(third_quartile)
    
    #agrupando o conjunto de erros para o tipo de dados - normal ou anormal
    for i in range(final_one_class_y_test.shape[0]):      
        if(final_one_class_y_test[i] == constants.ANOMALY_DATA_REPRESENTATION):
            error_test_anomaly.append(error[i])
        else:
            error_test_normal.append(error[i])
    
    
    # erro de teste da anomalia
    print('\nerror_test_anomaly')
    print(error_test_anomaly)
    print(len(error_test_anomaly))
    
    # variaveis que contem o erro de teste da anomalia
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.calc_error_metrics(error_test_anomaly)
    
    
    # erro de teste para dados normais
    print('\nerror_test_normal')
    print(error_test_normal) 
    print(len(error_test_normal))   
    
    # variaveis que contem o erro de teste para dados normais
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.calc_error_metrics(error_test_normal) 
    ''' end of new prints'''  
    
    
    # printando métricas para media do erro de reconstruçao da anomalia
    print('\nPrint informations - Mean informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_mean)
    
    
    # printando métricas para o segundo quartil do erro de reconstruçao da anomalia
    print('\nPrint informations - Third quartile informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_quartile)
    
    
    ################################################### plot do erro de reconstrução ##########################################################
    import matplotlib.pyplot as plt
    import numpy as np
    import pylab as P
    
    plt.figure(figsize =(20,15))
    
    qtd_normal = len(error_test_normal)
    qtd_anomaly = len(error_test_anomaly)
    
    plt.scatter(range(qtd_normal), error_test_normal, color = 'blue', label = str(qtd_normal) + ' normal data')
    plt.scatter(range(qtd_anomaly), error_test_anomaly, color = 'red', label = str(qtd_anomaly) + ' anomaly data')
    
    mean_normal = np.mean(error_test_normal)
    mean_anomaly = np.mean(error_test_anomaly)
    
    plt.axhline(y=mean_normal, linewidth=2, color = 'blue', label = 'normal mean: ' + str(mean_normal))
    plt.axhline(y=mean_anomaly, linewidth=2, color = 'red', label = 'anomaly mean: ' + str(mean_anomaly))
    
    plt.xlabel('instance number')
    plt.ylabel('logloss')
    
    plt.title(dataset_name + ' - reconstruction error')
    
    texto = (" mcc: %.2f | auc_roc:  %.2f | auc_prc: %.2f | f_score: %.2f  " %(mcc, auc_roc, auc_prc, f_score))
    plt.annotate(texto, xy=(0.45, 0.85), xytext=(0, 0), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', size=14, ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))
    
    plt.style.use('seaborn-poster')
    plt.legend()
    
    plt.savefig(dataset_name + '_err_recons.png')
    #plt.show()
    ########################################################################################################################################
    
    
    ################################################### plot do erro de reconstrução ordenado ##############################################
    
    plt.figure(figsize =(20,15))
    
    sort_error_test_normal = np.sort(error_test_normal)
    sort_error_test_anomaly = np.sort(error_test_anomaly)
    
    qtd_normal = len(error_test_normal)
    qtd_anomaly = len(error_test_anomaly)
    
    plt.scatter(range(qtd_normal), sort_error_test_normal, color = 'blue', label = str(qtd_normal) + ' normal data')
    plt.scatter(range(qtd_anomaly), sort_error_test_anomaly, color = 'red', label = str(qtd_anomaly) + ' anomaly data')
    
    plt.xlabel('instance number')
    plt.ylabel('logloss')
    
    plt.title(dataset_name + ' - order reconstruction error')
    
    texto = (" mcc: %.2f | auc_roc:  %.2f | auc_prc: %.2f | f_score: %.2f  " %(mcc, auc_roc, auc_prc, f_score))
    plt.annotate(texto, xy=(0.45, 0.85), xytext=(0, 0), xycoords=('axes fraction', 'figure fraction'), textcoords='offset points', size=14, ha='center', va='bottom', bbox=dict(boxstyle="round", fc="w", ec="0", alpha=1))
    
    plt.style.use('seaborn-poster')
    plt.legend()
    
    plt.savefig(dataset_name + '_sort_err_recons.png')
    #plt.show()
    ########################################################################################################################################
    
    
    ################################################### plot do histograma #################################################################
    
    plt.figure(figsize =(20,15))
    
    # anomaly
    mu_anomaly = np.mean(error_test_anomaly)
    sigma_anomaly = np.std(error_test_anomaly)
    n, bins, patches = P.hist(error_test_anomaly, 50, normed=1, histtype='stepfilled', label = 'anomaly', alpha = 0.5)
    n1 = np.max(n)
    
    # normal
    mu_normal = np.mean(error_test_normal)
    sigma_normal = np.std(error_test_normal)
    n, bins, patches = P.hist(error_test_normal, 50, normed=1, histtype='stepfilled', label = 'normal', alpha = 0.5)
    n2 = np.max(n)
    
    max = np.max([n1, n2]) + 0.1
    P.ylim([0, max])
    
    plt.title(dataset_name + ' - histogram')
    plt.legend()
    
    plt.savefig(dataset_name + '_histogram.png')
    #plt.show()
    ########################################################################################################################################
 
    return y_pred_anomaly_data_quartile

def CAE_anomaly_detection_10_run(X_train, Y_train, X_test, Y_test, dataset_name, num_classes, batch_normalization) :
    list_mcc, list_auc_roc, list_auc_prc, list_f_score, list_alpha = [], [], [], [], []
    list_quartile_mcc, list_quartile_auc_roc, list_quartile_auc_prc, list_quartile_f_score, list_quartile_alpha = [], [], [], [], []
    for i in range(10):
        if(batch_normalization):
            if(dataset_name == 'MNIST'):
                name_model = 'CAE_BN_MNIST_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_index' + str(i)+'.h5'
            elif(dataset_name == 'CIFAR'):
                name_model = 'CAE_BN_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_index' + str(i)+'.h5'
            elif(dataset_name == 'CIFAR100'):
                name_model = 'CAE100_BN_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_index' + str(i)+'.h5'
            print('\nModel: ' + name_model)
        
            try:
                print('Load..')    
                autoencoder = load_model(name_model)
            except:
                print('TRAINING..')
                if(constants.DEEP == 1):
#                     autoencoder = models.convnet_autoencoder_deep1_bn_invertido(X_train, dataset_name)
                    autoencoder = models.convnet_autoencoder_deep1_bn(X_train, dataset_name)
#                     autoencoder = models.convnet_autoencoder_deep1_augmentation(X_train, dataset_name)
#                     autoencoder = models.convnet_autoencoder_deep1_bn_augmentation(X_train, dataset_name)
#                     autoencoder = models.convnet_autoencoder_deep1_dropout(X_train, dataset_name)
#                 elif(constants.DEEP == 3):
#                     autoencoder = models.convnet_autoencoder_deep3_bn(X_train, dataset_name)
#                     autoencoder = models.convnet_autoencoder_deep3_bn_augmentation(X_train, dataset_name)
#                 elif(constants.DEEP == 5):
#                     autoencoder = models.convnet_autoencoder_deep5_bn(X_train, dataset_name)
#                     autoencoder = models.convnet_autoencoder_deep5_bn_augmentation(X_train, dataset_name)
                elif(constants.DEEP == 10):
#                     autoencoder = models.convnet_autoencoder_deep10_bn_invertido(X_train, dataset_name)
                    autoencoder = models.convnet_autoencoder_deep10_bn(X_train, dataset_name)
#                     autoencoder = models.convnet_autoencoder_deep10_augmentation(X_train, dataset_name)
#                     autoencoder = models.convnet_autoencoder_deep10_bn_augmentation(X_train, dataset_name)
#                     autoencoder = models.convnet_autoencoder_deep10_dropout(X_train, dataset_name)
                autoencoder.save(name_model)
        else:
            if(dataset_name == 'MNIST'):
                name_model = 'CAE_MNIST_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_index' + str(i)+'.h5'
            elif(dataset_name == 'CIFAR'):
                name_model = 'CAE_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_index' + str(i)+'.h5'
            elif(dataset_name == 'CIFAR100'):
                name_model = 'CAE_CIFAR100_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) + '_index' + str(i)+'.h5'
            print('\nModel: ' + name_model)
        
            try:
                print('Load..')    
                autoencoder = load_model(name_model)
            except:
                print('TRAINING..')
                if(constants.DEEP == 1):
                    autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep1(X_train, dataset_name)
                elif(constants.DEEP == 3):
                    autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep3(X_train, dataset_name)
                elif(constants.DEEP == 5):
                    autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep5(X_train, dataset_name)
                elif(constants.DEEP == 10):
                    autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep10(X_train, dataset_name)
#                     autoencoder, encoded, decoded, input_img = models.convdeconv_autoencoder_deep10(X_train, dataset_name)
                    
                autoencoder.save(name_model)
        
        print('PRED_TRAIN..')
        decoded_train = autoencoder.predict(X_train)
                
        # predict
        print('PRED..')
        decoded_imgs = autoencoder.predict(X_test)
        
#         util.save_img_CIFAR(X_test, decoded_imgs, name)
        
        print('GRIDSEARCH..')
        er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.gridsearch_error_logloss(X_train, decoded_train)
        ''' best_alpha'''
        alpha_mean = er_mean
        alpha_quartile = third_quartile
        
        '''anomaly detection'''
        print('\nMean: ')
        y_pred_anomaly_data_mean = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_mean)
        
#         print('Len y_pred_anomaly_data_mean:' + str(y_pred_anomaly_data_mean.shape) )
#         
#         unique, counts = np.unique(y_pred_anomaly_data_mean, return_counts=True)
#         temp = dict(zip(unique, counts))
#         print('y_pred_anomaly_data_mean')
#         print(temp)
        
        print('\nThird quartile: ')
        y_pred_anomaly_data_quartile = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_quartile)
        
#         print('Len y_pred_anomaly_data_mean:' + str(y_pred_anomaly_data_mean.shape) )
#         
#         unique, counts = np.unique(y_pred_anomaly_data_quartile, return_counts=True)
#         temp = dict(zip(unique, counts))
#         print('y_pred_anomaly_data_quartile')
#         print(temp)
    
        ''' one_class representation of the Y_test''' 
        final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)        
        
        print('\nPrint informations - Mean informations:')
        ''' both arrays must be in one_class form '''
        mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_mean)
#         print('\nPrint informations - Mean informations:')
#         mcc, auc_roc, auc_prc, f_score = util.print_information_temp(Y_test, y_pred_anomaly_data_mean)
        
        print('\nPrint informations - Third quartile informations:')
        ''' both arrays must be in one_class form '''
        mcc_quartile, auc_roc_quartile, auc_prc_quartile, f_score_quartile = util.print_information(final_one_class_y_test, y_pred_anomaly_data_quartile)
#         print('\nPrint informations - Third quartile informations:')
#         mcc_quartile, auc_roc_quartile, auc_prc_quartile, f_score_quartile = util.print_information_temp(Y_test, y_pred_anomaly_data_quartile)
        list_mcc.append(mcc)
        list_auc_roc.append(auc_roc)
        list_auc_prc.append(auc_prc)
        list_f_score.append(f_score)
        list_alpha.append(alpha_mean)
        
        list_quartile_mcc.append(mcc_quartile)
        list_quartile_auc_roc.append(auc_roc_quartile)
        list_quartile_auc_prc.append(auc_prc_quartile)
        list_quartile_f_score.append(f_score_quartile)
        list_quartile_alpha.append(alpha_quartile)
    print('\nMean lists:')
    util.print_lists(list_mcc, list_auc_roc, list_auc_prc, list_f_score, list_alpha)
    util.print_statistics(list_mcc, list_auc_roc, list_auc_prc, list_f_score)
    
    print('\nThird quartile lists:')
    util.print_lists(list_quartile_mcc, list_quartile_auc_roc, list_quartile_auc_prc, list_quartile_f_score, list_quartile_alpha)
    util.print_statistics(list_quartile_mcc, list_quartile_auc_roc, list_quartile_auc_prc, list_quartile_f_score)
 
    return y_pred_anomaly_data_quartile
    
def binary_CAE_anomaly_detection(X_train, Y_train, X_test, Y_test, dataset_name, num_classes):
    y_preds, mean_preds = [], []
    for i in range(num_classes):
        if(i == constants.ANOMALY_CLASS):
            continue
        
        if(dataset_name == 'MNIST'):
            name = 'Binary_CAEs_MNIST_' + str(constants.OPT_FUNC) +'_without_' + str(i) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)
        elif(dataset_name == 'CIFAR'):
            name = 'Binary_CAEs_CIFAR_' + str(constants.OPT_FUNC) +'_without_' + str(i) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH) 
        elif(dataset_name == 'CIFAR100'):
            name = 'Binary_CAEs_CIFAR100_' + str(constants.OPT_FUNC) +'_without_' + str(i) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)
        name_model = name +'.h5'
        print('\nModel: ' + name_model)
        
        print('INDEX: ' + str(i))
        _, _, x_class, y_class = util.pre_processing_X_train_multiclass(X_train, Y_train, i)
         
        try:
            print('Load..')    
            autoencoder = load_model(name_model)
        except:
            print('TRAINING..')
            autoencoder, encoded, decoded, input_img = models.cae_flatten(x_class, dataset_name)
            ''' antigo '''
#             if(constants.DEEP == 1):
#                 autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep1(x_class, dataset_name)
# #                 autoencoder, encoded, decoded, input_img = convnet_autoencoder_deep1(x_class)
#             elif(constants.DEEP == 3):
#                 autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep3(x_class, dataset_name)
# #                 autoencoder, encoded, decoded, input_img = convnet_autoencoder_deep3(x_class)
#             elif(constants.DEEP == 5):
#                 autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep5(x_class, dataset_name)
# #                 autoencoder, encoded, decoded, input_img = convnet_autoencoder_deep5(x_class)
#             elif(constants.DEEP == 10):
#                 autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep10(x_class, dataset_name)    
# #                 autoencoder, encoded, decoded, input_img = convnet_autoencoder_deep10(x_class)            
            autoencoder.save(name_model)
        
        unique, counts = np.unique(y_class, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter')
        print(temp)
        
        print('PRED_TRAIN..')
        decoded_train = autoencoder.predict(x_class)
        
#         name_train = name + '_train'
#         util.save_img_CIFAR(x_class, decoded_train, name_train)
        # predict
        print('PRED..')
        decoded_imgs = autoencoder.predict(X_test)
        
#         name_test = name + '_test'
#         util.save_img_CIFAR(X_test, decoded_imgs, name_test)
         
        print('GRIDSEARCH..')
        er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.gridsearch_error_logloss(x_class, decoded_train)
        ''' best_alpha'''
        alpha_mean = er_mean
        alpha_quartile = third_quartile
         
        '''anomaly detection'''
        print('\nMean: ')
        y_pred_anomaly_data_mean = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_mean)
        print('\nThird quartile: ')
        y_pred_anomaly_data_quartile = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_quartile)
        
        ''' appending preds'''
        y_preds.append(y_pred_anomaly_data_quartile)
        mean_preds.append(y_pred_anomaly_data_mean)
    
    compiled_ypreds = util.get_compiled_ypreds(Y_test, y_preds, num_classes)
    compiled_mean_preds = util.get_compiled_ypreds(Y_test, mean_preds, num_classes)
    
    print('compiled_ypreds')
    print(compiled_ypreds)
    temp = np.asarray(compiled_ypreds)
    print(temp.shape)
    
    print('compiled_mean_preds')
    print(compiled_mean_preds)
    temp = np.asarray(compiled_mean_preds)
    print(temp.shape)
    
    ''' one_class representation of the Y_test''' 
    final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)        
    
    print('\nPrint informations - Mean informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, compiled_mean_preds)
    
    print('\nPrint informations - Third quartile informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, compiled_ypreds)
    
    return compiled_ypreds
    
def OCSVM_anomaly_detection(X_train, Y_train, X_test, Y_test, clf_type, dataset_name):
    unique, counts = np.unique(Y_train, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_one_class_y_train')
    print(temp)
        
    if(clf_type == 'DEF_OCSVM'):
        if(dataset_name == 'MNIST'):
            name_model = 'MNIST_def_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        elif(dataset_name == 'CIFAR'):
            name_model = 'CIFAR_def_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        elif(dataset_name == 'CIFAR100'):
            name_model = 'CIFAR100_def_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        elif(dataset_name == 'SI'):
            name_model = 'SI_def_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        elif(dataset_name == 'COIL20'):
            name_model = 'COIL20_def_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        print('\nModel: ' + name_model)
        
        try:
            print('LOAD...')
            clf = joblib.load(name_model)
        except:    
        
            unique, counts = np.unique(Y_train, return_counts=True)
            temp = dict(zip(unique, counts))
            print('Y_train')
            print(temp)
            
            print('TRAINING...')
            clf = svm.OneClassSVM()
            clf = clf.fit(X_train)
              
            print('SAVE...')
            joblib.dump(clf, name_model)
            
            
    elif(clf_type == 'GRID_OCSVM'):   
        if(dataset_name == 'MNIST'):
            name_model = 'MNIST_grid_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        elif(dataset_name == 'CIFAR'):
            name_model = 'CIFAR_grid_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        elif(dataset_name == 'SI'):
            name_model = 'SI_grid_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        elif(dataset_name == 'COIL20'):
            name_model = 'COIL20_grid_ocsvm_anomaly_' + str(constants.ANOMALY_CLASS)+'.pkl'
        print('\nModel: ' + name_model)
        
        Z_train = np.ones((Y_train.shape[0],), dtype=np.int)
        
        try:
            print('LOAD...')
            clf = joblib.load(name_model)

        except:
            #param_grid = {'nu': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            #'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, gamma_auto], }
            #clf = GridSearchCV(OneClassSVM(kernel='rbf'), param_grid, scoring='f1')
            
            print('TRAINING...')
            grid_clf = GridSearchCV(OneClassSVM(kernel='rbf'), cv=3, n_jobs=-1, param_grid = {'nu': [0.1, 0.2, 0.3, 0.4, 0.5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]}, scoring='f1')
                    #grid_clf = grid_clf.fit(X_train, Y_train)
            
            ''' only normal data, so, labels = 1'''
            grid_clf = grid_clf.fit(X_train, Z_train)
            
            clf = grid_clf.best_estimator_
            print('BEST')
            print(clf)
            
            print('SAVE...')
            joblib.dump(clf, name_model)
    
    print('CLASSIFIER: ')
    print(clf)
    
    print('TESTING..')
    clf_pred = clf.predict(X_test)
    print('clf_pred')
    print(clf_pred)
    
    unique, counts = np.unique(clf_pred, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_clf_pred')
    print(temp)
    
    for i in range(len(clf_pred)):
        if(clf_pred[i] == -1):
            clf_pred[i] = constants.ANOMALY_DATA_REPRESENTATION
     
    print('clf_pred-depois')
    print(clf_pred)
    
    unique, counts = np.unique(clf_pred, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_clf_pred')
    print(temp)

    print('y_test')
    print(Y_test)
    
    unique, counts = np.unique(Y_test, return_counts=True)
    temp = dict(zip(unique, counts))
    print('counter_Y_test')
    print(temp)
    
    if((dataset_name == 'MNIST')):
        print('\nPrint informations:')
        
        unique, counts = np.unique(Y_test, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_Y_test')
        print(temp)
        
        unique, counts = np.unique(clf_pred, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_clf_pred')
        print(temp)
        
        ''' both arrays must be in one_class form '''
        mcc, auc_roc, auc_prc, f_score = util.print_information(Y_test, clf_pred)
        #mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, clf_pred)
        
    elif((dataset_name == 'CIFAR') or (dataset_name == 'SI') or (dataset_name == 'COIL20')):
        final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)
        print('\nPrint informations:')
    
        unique, counts = np.unique(Y_test, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_Y_test')
        print(temp)
        
        unique, counts = np.unique(clf_pred, return_counts=True)
        temp = dict(zip(unique, counts))
        print('counter_clf_pred')
        print(temp)
        
        ''' both arrays must be in one_class form '''
        mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, clf_pred)
        
    return clf_pred

def CAE_temp(X_train, Y_train, X_test, Y_test, dataset_name, num_classes, name_model):
    try:
        print('Load..')    
        autoencoder = load_model(name_model)
    except:
        print('TRAINING..')
        if(constants.DEEP == 1):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep1_split_channel(X_train, dataset_name)
        elif(constants.DEEP == 3):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep3(X_train, dataset_name)
        elif(constants.DEEP == 5):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep5(X_train, dataset_name)
        elif(constants.DEEP == 10):
            autoencoder, encoded, decoded, input_img = models.convnet_autoencoder_deep10_split_channel(X_train, dataset_name)
        autoencoder.save(name_model)
    
    print('PRED_TRAIN..')
    decoded_train = autoencoder.predict(X_train)
    # predict
    print('PRED..')
    decoded_imgs = autoencoder.predict(X_test)
    
    return decoded_train, decoded_imgs
    
#     print('GRIDSEARCH..')
#     er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.gridsearch_error_logloss(X_train, decoded_train)
#     ''' best_alpha'''
#     alpha_mean = er_mean
#     alpha_quartile = third_quartile
#     
#     '''anomaly detection'''
#     print('\nMean: ')
#     y_pred_anomaly_data_mean = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_mean)
#     print('\nThird quartile: ')
#     y_pred_anomaly_data_quartile = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_quartile)
#     
#     ''' one_class representation of the Y_test''' 
#     final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)        
#     
#     print('\nPrint informations - Mean informations:')
#     ''' both arrays must be in one_class form '''
#     mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_mean)
#     
#     print('\nPrint informations - Third quartile informations:')
#     ''' both arrays must be in one_class form '''
#     mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_quartile)
#  
#     return y_pred_anomaly_data_quartile

def CAE_anomaly_detection_split_channels(X_train, Y_train, X_test, Y_test, dataset_name, num_classes):
    ''' SPLITTING CHANNELS'''
    red_Xtrain, green_Xtrain, blue_Xtrain = [],[],[]  
    red_Xtest, green_Xtest, blue_Xtest = [],[],[]
    print('\nSplitting Xtrain')
    for input_image in X_train:
        red, green, blue = util.get_image_channels(input_image)
        red_Xtrain.append(red)
        green_Xtrain.append(green)
        blue_Xtrain.append(blue)
    
    print('\nSplitting Xtest')
    for input_image in X_test:
        red, green, blue = util.get_image_channels(input_image)
        red_Xtest.append(red)
        green_Xtest.append(green)
        blue_Xtest.append(blue)
        
    red_Xtrain = np.asarray(red_Xtrain)
    green_Xtrain = np.asarray(green_Xtrain)
    blue_Xtrain = np.asarray(blue_Xtrain)
        
    red_Xtest = np.asarray(red_Xtest)
    green_Xtest = np.asarray(green_Xtest)
    blue_Xtest = np.asarray(blue_Xtest)
        
    print('Xtrain')
    print('red_Xtrain')
    print(red_Xtrain.shape)
    print('green_Xtrain')
    print(green_Xtrain.shape)
    print('blue_Xtrain')
    print(blue_Xtrain.shape)
    
    print('Xtest')
    print('red_Xtest')
    print(red_Xtest.shape)
    print('green_Xtest')
    print(green_Xtest.shape)
    print('blue_Xtest')
    print(blue_Xtest.shape)
    
    
    name_model = 'RED_CAE_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) +'.h5'
#     name_model = 'RED_CAE_CIFAR_adam_deep10_filter64_epoch10_anomaly0.h5'
    print('\nModel: ' + name_model)
    red_decoded_train, red_decoded_imgs = CAE_temp(red_Xtrain, Y_train, red_Xtest, Y_test, dataset_name, num_classes, name_model)
    
    name_model = 'GREEN_CAE_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) +'.h5'
#     name_model = 'GREEN_CAE_CIFAR_adam_deep10_filter64_epoch10_anomaly0.h5'
    print('\nModel: ' + name_model)
    green_decoded_train, green_decoded_imgs = CAE_temp(green_Xtrain, Y_train, green_Xtest, Y_test, dataset_name, num_classes, name_model)
    
    name_model = 'BLUE_CAE_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS) +'.h5'
#     name_model = 'BLUE_CAE_CIFAR_adam_deep10_filter64_epoch10_anomaly0.h5'
    print('\nModel: ' + name_model)
    blue_decoded_train, blue_decoded_imgs = CAE_temp(blue_Xtrain, Y_train, blue_Xtest, Y_test, dataset_name, num_classes, name_model)
    
    decoded_train = np.concatenate((red_decoded_train, green_decoded_train, blue_decoded_train), axis=3)
    
    print('decoded_train')
    #print(decoded_train)
    print(decoded_train.shape)
    
    decoded_imgs = np.concatenate((red_decoded_imgs, green_decoded_imgs, blue_decoded_imgs), axis=3)
    
    print('decoded_train')
    #print(decoded_train)
    print(decoded_train.shape)
    
#     util.save_img_CIFAR(X_test, decoded_imgs, 'imagem_teste')
    
    print('GRIDSEARCH..')
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.gridsearch_error_logloss(X_train, decoded_train)
    ''' best_alpha'''
    alpha_mean = er_mean
    alpha_quartile = third_quartile
    
    '''anomaly detection'''
    print('\nMean: ')
    y_pred_anomaly_data_mean = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_mean)
    print('\nThird quartile: ')
    y_pred_anomaly_data_quartile = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_quartile)
    
    ''' one_class representation of the Y_test''' 
    final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)        
    
    print('\nPrint informations - Mean informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_mean)
    
    print('\nPrint informations - Third quartile informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_quartile)
 
    return y_pred_anomaly_data_quartile
    
def denoising_CAE_anomaly_detection(X_train, Y_train, X_test, Y_test, X_train_noisy, X_test_noisy, dataset_name, num_classes):  
    ''' X_test: X_test_noisy'''      
#     if(dataset_name == 'MNIST'):
#         name = 'CAE_MNIST_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS)
#     elif(dataset_name == 'CIFAR'):
#         name = 'CAE_CIFAR_' + str(constants.OPT_FUNC) +'_deep' + str(constants.DEEP)+ '_filter' + str(constants.NB_FILTERS)+ '_epoch'+ str(constants.NB_EPOCH)+ '_anomaly'+ str(constants.ANOMALY_CLASS)
    ''' denoising '''
    name = 'denoising_no_pool_3l_64_CAE_CIFAR'
    name_model = name +'.h5'
    print('\nModel: ' + name_model)

    try:
        print('Load..')    
        autoencoder = load_model(name_model)
    except:
        print('TRAINING..')
        ''' denoising'''
        autoencoder, encoded, decoded, input_img = models.denoising_convnet_autoencoder_no_pool_layers(X_train, X_train_noisy, dataset_name)
        autoencoder.save(name_model)
    
    print('PRED_TRAIN..')
    decoded_train = autoencoder.predict(X_train_noisy)
#     name_train = name + '_train'
#     util.save_img_CIFAR(X_train, decoded_train, name_train)

    # predict
    print('PRED..')
    decoded_imgs = autoencoder.predict(X_test_noisy)    
#     name_test = name + '_test'
#     util.save_img_CIFAR(X_test, decoded_imgs, name_test)
    
    print('GRIDSEARCH..')
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.gridsearch_error_logloss(X_train, decoded_train)
    ''' best_alpha'''
    alpha_mean = er_mean
    alpha_quartile = third_quartile
    
    '''anomaly detection'''
    print('\nMean: ')
    y_pred_anomaly_data_mean = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_mean)
    print('\nThird quartile: ')
    y_pred_anomaly_data_quartile = util.anomaly_detection_logloss(X_test, decoded_imgs, alpha_quartile)
    
    ''' one_class representation of the Y_test''' 
    final_one_class_y_test = util.pre_processing_one_class_label(Y_test, constants.ANOMALY_CLASS)      
    
    ''' new prints for test'''
    print('final_one_class_y_test')
    print(final_one_class_y_test)
    print('shape[0]')
    print(final_one_class_y_test.shape[0])
    
    error_test_normal = []
    error_test_anomaly = []
    
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile, error = util.gridsearch_error_logloss_debug(X_test, decoded_imgs)
    
    print('\nTEST')
    print('er_mean')
    print(er_mean)
    print('er_min')
    print(er_min)
    print('er_max')
    print(er_max)
    print('first_quartile')
    print(first_quartile)
    print('second_quartile')
    print(second_quartile)
    print('third_quartile')
    print(third_quartile)
#     print('\nERROR')
#     print(error)
    
    for i in range(final_one_class_y_test.shape[0]):      
        if(final_one_class_y_test[i] == constants.ANOMALY_DATA_REPRESENTATION):
            error_test_anomaly.append(error[i])
        else:
            error_test_normal.append(error[i])
    
    print('\nerror_test_anomaly')
    print(error_test_anomaly)
    print(len(error_test_anomaly))
    
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.calc_error_metrics(error_test_anomaly)
    
    print('\nerror_test_normal')
    print(error_test_normal) 
    print(len(error_test_normal))   
    
    er_mean, er_min, er_max, first_quartile, second_quartile, third_quartile = util.calc_error_metrics(error_test_normal) 
    ''' end of new prints'''  
    
    print('\nPrint informations - Mean informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_mean)
    
    print('\nPrint informations - Third quartile informations:')
    ''' both arrays must be in one_class form '''
    mcc, auc_roc, auc_prc, f_score = util.print_information(final_one_class_y_test, y_pred_anomaly_data_quartile)
 
    return y_pred_anomaly_data_quartile    