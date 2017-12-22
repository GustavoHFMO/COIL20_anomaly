#!/usr/local/bin/python

import util, constants, detection_and_classification
import scipy.io as sio

dataset_name = 'COIL20'
SVHN_num_classes = 20

# clf_type = ['CNN', 'CAE', 'DEF_OCSVM', 'GRID_OCSVM']
clf_type = 'CAE'

    
if __name__ == '__main__':
    
    # The data, shuffled and split between train and test sets:
    # carregando os datasets
    train_data = sio.loadmat('COIL20.mat')
    
    # importando os dados de treinamento
    x_train = train_data['fea']
    y_train = train_data['gnd']
    
    #x_train = x_train[:100]
    #y_train = y_train[:100]
    
    # pre processando os dados
    X_train, Y_train, X_test, y_test = util.pre_processing_coil20(x_train, y_train, clf_type)
    
      
    #RUNNING ANOMALY DETECTION OPERATION
    ''' 
    if multi_binary_clfs == True: trains a binary_clf for each class
    if multi_binary_clfs == False: trains a clf for all class, except the anomaly class
    '''
    multi_binary_clfs = False
    batch_normalization = False
    dropout = False
    
    
    if(multi_binary_clfs):
        if (clf_type=='CAE'):
            compiled_ypreds = detection_and_classification.binary_CAE_anomaly_detection(X_train, Y_train, X_test, y_test, dataset_name, SVHN_num_classes)
        elif(clf_type=='CNN'):
            compiled_ypreds = detection_and_classification.binary_cnns_anomaly_detection(X_train, Y_train, X_test, y_test, dataset_name, SVHN_num_classes)
        else: #DEF_OCSVM, GRID_OCSVM
            compiled_ypreds = detection_and_classification.multi_ocsvm_anomaly_detection(X_train, Y_train, X_test, y_test, clf_type, dataset_name, SVHN_num_classes)
    else:
        if(clf_type=='CAE'):
            
            executions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            for i in range(executions):
                compiled_ypreds = detection_and_classification.CAE_anomaly_detection(X_train, Y_train, X_test, y_test, dataset_name, SVHN_num_classes, i)
            
            #i = 9
            #compiled_ypreds = detection_and_classification.CAE_anomaly_detection(X_train, Y_train, X_test, y_test, dataset_name, SVHN_num_classes, i)
                
                
        elif(clf_type=='CNN'):
            compiled_ypreds = detection_and_classification.multiclass_cnn_anomaly_detection(X_train, Y_train, X_test, y_test, dataset_name, SVHN_num_classes, batch_normalization, dropout)
        else: #DEF_OCSVM, GRID_OCSVM
            compiled_ypreds = detection_and_classification.OCSVM_anomaly_detection(X_train, Y_train, X_test, y_test, clf_type, dataset_name)
            
    ''' PERFORM CLASSIFICATION '''
    perform_classification = False
    if perform_classification:
        #Reshaping data to cnn appropriate form
        if ((clf_type=='DEF_OCSVM') or (clf_type=='GRID_OCSVM')):
            X_train, Y_train = util.pre_processing_CIFAR_data_without_class(x_train, y_train, constants.ANOMALY_CLASS, True)
        #Classification:
        print('\nPERFORMING CLASSIFICATION:')
        detection_and_classification.multiclass_classification(X_train, Y_train, X_test, y_test, compiled_ypreds, dataset_name, SVHN_num_classes)
    
    ''' END '''                
    
        