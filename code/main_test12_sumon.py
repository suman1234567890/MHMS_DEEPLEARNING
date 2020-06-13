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
    for time in range(tot_iter):
        idx_test = idx_test + 1
        X_train_plot = np.mean(X_train, axis=1).flatten()
        X_test_plot = np.mean(X_test, axis=1).flatten()
        linear_svr = build_SVR('linear', 1000)
        print(X_train)
        print(X_test)
        print(y_train)
        print(Y_test)

        linear_svr.fit(X_train, y_train)
        y_pred = linear_svr.predict(X_test)

        y_pred.reshape(1, X_test.shape[0])

        # logging.info(" %f  linear predicted result: %s" %(idx_test, y_pred))

        if y_predTotal.shape[0] < 1:
            y_predTotal = y_pred
        else:
            y_predTotal = np.append(y_predTotal, y_pred, axis=0)

        # plt.scatter(y_train, y_train, color='blue')
        # plt.show()
    # plt.plot(X_train, y_train,  color='red', linewidth=2)
    # plt.plot(X_test, y_predTotal.flatten(), color='blue', linewidth=2,label='linear')

    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='blue', linewidth=2,
             label='linear')
    plt.legend()

    # plt.savefig('images/linear.png')

    y_predTotal = np.reshape(y_predTotal, (tot_iter, X_test.shape[0]))

    # logging.info("  linears   result: %s" % (y_predTotal))

    logging.info('  linears predicted mean result: %s'
                 % y_predTotal.mean(axis=0))

    # score= mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)
    # logging.info("   linear svr result: %f_%f" %(score, mae_score))

    y_predTotal = np.array([])
    idx_test = 0
    for time in range(tot_iter):
        idx_test = idx_test + 1
        NUM_ESTIMATOR = 50
        NUM_PREEPOCH = 150
        NUM_BPEPOCH = 175
        BATH_SIZE = 50
        rf = build_RF(NUM_ESTIMATOR)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_pred.reshape(1, X_test.shape[0])

        # logging.info(" r fores  predict  result: %s" %(y_pred))

        if y_predTotal.shape[0] < 1:
            y_predTotal = y_pred
        else:
            y_predTotal = np.append(y_predTotal, y_pred, axis=0)
    y_predTotal = np.reshape(y_predTotal, (tot_iter, X_test.shape[0]))

    # plt.scatter(y_train, y_train, color='blue')
    # plt.show()

    # plt.clf()
    # plt.plot(X_train, y_train,  color='red', linewidth=2)
    # plt.plot(X_test, y_predTotal.flatten(), color='blue', linewidth=2)

    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='red', linewidth=2,
             label='random forest')
    plt.legend()
    t1=list(range(0, len(y_predTotal.flatten())))
    t2=y_predTotal.flatten()
    t=[]
    for a in t1:
        for b in t2:
            t.append((a,b))

    # plt.savefig('images/rforest.png')

    logging.info('  r fore predicted mean result: %s'
                 % y_predTotal.mean(axis=0))
    savetxt('dataRF.csv', t, delimiter=',') 
    # score= mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)
    # logging.info("   randomforest  result: %f_%f" %(score, mae_score))
    # neural network

    y_predTotal = np.array([])
    idx_test = 0
    for time in range(tot_iter):
        idx_test = idx_test + 1
        nn_model = build_NN(data_dim)
        sc = StandardScaler()
        nn_model.fit(X_train, y_train, epochs=50, batch_size=BATH_SIZE)
        y_pred = nn_model.predict(X_test)
        y_predact = y_pred
        y_pred.reshape(1, X_test.shape[0])

        # logging.info(" neu  predict  result: %s" %(y_pred))

        if y_predTotal.shape[0] < 1:
            y_predTotal = y_predact
        else:
            y_predTotal = np.append(y_predTotal, y_predact, axis=0)
    y_predTotal = np.reshape(y_predTotal, (tot_iter, X_test.shape[0]))

    # plt.clf()
    # plt.plot(X_train, y_train,  color='red', linewidth=2)
    # plt.plot(X_test, y_predTotal.flatten(), color='green', linewidth=2)

    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='green', linewidth=2,
             label='Neural')
    plt.legend()

    # plt.savefig('images/neural.png')

    logging.info('  neu predicted mean result: %s'
                 % y_predTotal.mean(axis=0))

    idx_test = 0
    y_predTotal = np.array([])
    for time in range(tot_iter):
        idx_test = idx_test + 1

        normal_AE = build_pre_normalAE(data_dim, X_train,
                epoch_pretrain=NUM_PREEPOCH, hidDim=[140, 280])
        normal_AE.fit(X_train, y_train, epochs=NUM_BPEPOCH,
                      batch_size=BATH_SIZE)
        y_pred = normal_AE.predict(X_test)
        y_pred.reshape(1, X_test.shape[0])

        # logging.info(" ae  predict  result: %s" %(y_pred))

        if y_predTotal.shape[0] < 1:
            y_predTotal = y_pred
        else:
            y_predTotal = np.append(y_predTotal, y_pred, axis=0)
    y_predTotal = np.reshape(y_predTotal, (tot_iter, X_test.shape[0]))

    # plt.clf()
    # plt.plot(X_train, y_train,  color='red', linewidth=2)
    # plt.plot(X_test, y_predTotal.flatten(), color='red',  label='AE')

    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='red', linewidth=2,
             label='auto encoder')
    plt.legend()

    # plt.savefig('images/ae.png')

    logging.info('  ae predicted mean result: %s'
                 % y_predTotal.mean(axis=0))

    # score = mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)
    # logging.info("   normal ae result: %f_%f" %(score, mae_score))
    # denoise AE

    idx_test = 0
    y_predTotal = np.array([])
    for time in range(tot_iter):
        idx_test = idx_test + 1
        denois_AE = build_pre_denoiseAE(data_dim, X_train,
                epoch_pretrain=NUM_PREEPOCH, hidDim=[140, 280])
        denois_AE.fit(X_train, y_train, epochs=NUM_BPEPOCH,
                      batch_size=BATH_SIZE)
        y_pred = denois_AE.predict(X_test)
        y_pred.reshape(1, X_test.shape[0])

        # logging.info(" denoiseae  predict  result: %s" %(y_pred))

        if y_predTotal.shape[0] < 1:
            y_predTotal = y_pred
        else:
            y_predTotal = np.append(y_predTotal, y_pred, axis=0)
    y_predTotal = np.reshape(y_predTotal, (tot_iter, X_test.shape[0]))

    # plt.clf()
    # plt.plot(X_train, y_train,  color='red', linewidth=2)
    # plt.plot(X_test, y_predTotal.flatten(), color='cyan', linewidth=2)

    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='cyan', linewidth=2,
             label='Denoise AE')
    plt.legend()

    # plt.savefig('images/denoiseAE.png')

    logging.info('  denoiseae predicted mean result: %s'
                 % y_predTotal.mean(axis=0))

    # score = mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)
    # logging.info("   denoise ae result: %f_%f" %(score, mae_score))
    # rbf using sigmoid function, feature should be scaled to -1 and 1

    idx_test = 0

    for time in range(tot_iter):
        idx_test = idx_test + 1

        scaler = MinMaxScaler()
        X_train_rbm = scaler.fit_transform(X_train)
        rbm = build_RBM(NUM_BPEPOCH, NUM_PREEPOCH, batch_size=BATH_SIZE)
        rbm.fit(X_train_rbm, y_train)
        X_test_rbm = scaler.transform(X_test)
        y_pred = rbm.predict(X_test_rbm)
        y_predTotal = np.reshape(y_pred, (tot_iter, X_test.shape[0]))
        logging.info('  dbn predict  result: %s' % y_pred)

    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='#808000', linewidth=2,
             label='dbn')
    plt.legend()
    
        # plt.plot(X_test, y_predTotal.flatten(), color='magenta', linewidth=2)
        # plt.savefig('images/biLSTM.png')

        # score = mean_squared_error(y_pred, y_test)
        # mae_score = mean_absolute_error(y_pred, y_test)
        # logging.info("   dbn result: %f_%f" %(score, mae_score))
        # Bi-directional LSTM
        # t = load_data(False)

    X_train_init = X_train
    X_test_init = X_test

    # X_train = np.reshape(X_train, ( 1,X_train.shape[1],X_train.shape[0]))

    data_dim = X_train.shape[1]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    timesteps = 1
   
    #data_dim = X_train.shape[2]
    #timesteps = X_train.shape[0]

    biLSTM = build_BILSTM(timesteps, data_dim)
    biLSTM.fit(X_train, y_train, epochs=NUM_BPEPOCH,
               batch_size=BATH_SIZE)
    y_pred = biLSTM.predict(X_test)

    y_predTotal = np.reshape(y_pred, (tot_iter, X_test.shape[0]))
    logging.info('predicted BILSTM mean result: %s'
                 % y_predTotal.mean(axis=0))

    # plt.clf()
    # plt.plot(X_train, y_train,  color='red', linewidth=2)

    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='yellow', linewidth=2,
             label='BILSTM')
    plt.legend()
    

    # plt.plot(X_test, y_predTotal.flatten(), color='magenta', linewidth=2)
    # plt.savefig('images/biLSTM.png')

    # score = mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)
    # logging.info("   bi-lstm result: %f_%f" %(score, mae_score))
    # LSTM
    
    LSTM = build_LSTM(timesteps, data_dim)
    LSTM.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
    y_pred = LSTM.predict(X_test)

    # score = mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)

    y_predTotal = np.reshape(y_pred, (tot_iter, X_test.shape[0]))
    logging.info('predicted mean result: %s' % y_predTotal.mean(axis=0))

    # plt.clf()
    # plt.plot(X_train_plot, y_train,  color='red', linewidth=2)
    # plt.plot(X_test_plot, y_predTotal.flatten(), color='blue', linewidth=2)

    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='blue', linewidth=2,
             label='LSTM')
    plt.legend()
    
    # plt.savefig('images/LSTM.png')

    # CNN
    X_train = X_train_init
    X_test = X_test_init 

    X_train = X_train.reshape(50, 1, 1)
    X_test = X_test.reshape(25, 1, 1)

    CNN = build_CNN(1, 1)
    CNN.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
    y_pred = CNN.predict(X_test)
    savetxt('dataLSTM.csv', y_predTotal, delimiter=',')

 
    #plt.clf()
    #plt.plot(X_train_plot, y_train, color='red', linewidth=2)
    #plt.plot(X_test_plot, y_pred.flatten(), color='blue', linewidth=2)
    y_predTotal = np.reshape(y_pred, (tot_iter, X_test.shape[0]))
    logging.info('predicted mean result: %s' % y_predTotal.mean(axis=0))
    #
    plt.clf()
    y_predTotal = preprocessing.normalize(y_predTotal)
    plt.plot(list(range(0, len(y_predTotal.flatten()))),
             y_predTotal.flatten(), color='green', linewidth=2,
             label='CNN')
    plt.legend()
    plt.savefig('images/combo.png')
    savetxt('dataCNN.csv', y_predTotal, delimiter=',')

 

if __name__ == '__main__':
    t = LoadIt3105()

    mains(t)

    # t=LoadIt0604(2)
    # mains(t)
