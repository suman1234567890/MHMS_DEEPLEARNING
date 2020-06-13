#!/usr/bin/python
# -*- coding: utf-8 -*-
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from numpy import savetxt



from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from data_loaders_csv import *
from feature_extract import gen_fea
from sklearn.preprocessing import MinMaxScaler
import logging
from normal_models import build_SVR, build_RF, build_NN
from dl_models import build_BILSTM, build_LSTM, build_CNN, \
    build_pre_normalAE, build_pre_denoiseAE, build_RBM
from data_loaders import load_data
from numpy.random import seed
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

seed(10132017)

# import tensorflow
# tensorflow.random.set_seed(18071991)

log = 'output.log'
logging.basicConfig(filename=log, level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')


def mains(t):
    (X_train, y_train, X_test, Y_test) = (t[0], t[1], t[2], t[3])
    X_train= np.reshape(t[0], (-1, 1))
    X_test =np.reshape(t[2], (-1, 1))
    data_dim = X_train.shape[1]
    idx_test = 0
    tot_iter = 1
    plt.plot(list(range(0, len(Y_test.flatten()))), Y_test.flatten(),
             color='Black', linewidth=2, label='Actual')
    y_predTotal = np.array([])
    idx_test = 0
    NUM_ESTIMATOR = 50
    NUM_PREEPOCH = 150
    NUM_BPEPOCH = 175
    BATH_SIZE = 50

    #Linear section started

    X_train_plot = np.mean(X_train, axis=1).flatten()
    X_test_plot = np.mean(X_test, axis=1).flatten()
    linear_svr = build_SVR('linear', 1000)
    linear_svr.fit(X_train, y_train)
    y_pred = linear_svr.predict(X_test)
    y_pred.reshape(1, X_test.shape[0])
    savetxt('outputcsv/linear.csv', y_pred, delimiter=',')
    plt.plot(list(range(0, len(y_pred.flatten()))),
             y_pred.flatten(), color='red', linewidth=2,
             label='random forest')
    #Linear section Ended

    # Random Forest Section started

    rf = build_RF(NUM_ESTIMATOR)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred.reshape(1, X_test.shape[0])
    plt.plot(list(range(0, len(y_pred.flatten()))),
             y_pred.flatten(), color='red', linewidth=2,
             label='random forest')
    savetxt('outputcsv/dataRF.csv', y_pred, delimiter=',') 
    plt.legend()
    plt.savefig('images/RandomForest.png')
    
    # Random Forest Section End

    #Neural Method Start


    nn_model = build_NN(data_dim)
    sc = StandardScaler()
    nn_model.fit(X_train, y_train, epochs=50, batch_size=BATH_SIZE)
    y_pred = nn_model.predict(X_test)
    y_predact = y_pred
    y_pred.reshape(1, X_test.shape[0])

    plt.plot(list(range(0, len(y_pred.flatten()))),
             y_pred.flatten(), color='red', linewidth=2,
             label='neural')
    savetxt('outputcsv/Meural.csv', y_pred, delimiter=',') 
    plt.legend()
    plt.savefig('images/Neural.png')
    

    #Neural Method End
 

if __name__ == '__main__':
    t = LoadIt3105()

    mains(t)

    # t=LoadIt0604(2)
    # mains(t)
