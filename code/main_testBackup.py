from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from data_loaders_csv import *
from feature_extract import gen_fea
from sklearn.preprocessing import MinMaxScaler
import logging
from normal_models import build_SVR, build_RF, build_NN
from dl_models import build_BILSTM, build_LSTM, build_CNN, build_pre_normalAE, build_pre_denoiseAE, build_RBM
from data_loaders import load_data
from numpy.random import seed
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed(10132017)

# import tensorflow
# tensorflow.random.set_seed(18071991)


log = "output.log"
logging.basicConfig(filename=log, level=logging.DEBUG,
                    format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')


def mains(t):
    X_train, y_train, X_test = t[0], t[1], t[2]
    data_dim = X_train.shape[1]

    logging.info("X_Train: %s" % (X_train))
    logging.info("Y_Train: %s" % (y_train))
    logging.info("X_test: %s" % (X_test))
    idx_test = 0
    tot_iter = 1
    y_predTotal = np.array([])

    # score= mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)
    # logging.info("   linear svr result: %f_%f" %(score, mae_score))

    y_predTotal = np.array([])
    idx_test = 0

    idx_test = idx_test+1
    NUM_ESTIMATOR = 50
    NUM_PREEPOCH = 150
    NUM_BPEPOCH = 175
    BATH_SIZE = 50
   
    # score = mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)
    # logging.info("   denoise ae result: %f_%f" %(score, mae_score))
    # rbf using sigmoid function, feature should be scaled to -1 and 1





    # score = mean_squared_error(y_pred, y_test)
    # mae_score = mean_absolute_error(y_pred, y_test)
    # logging.info("   bi-lstm result: %f_%f" %(score, mae_score))
    # LSTM
    features_set = X_train
    test_features = X_test

    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

    LSTM = build_LSTM(features_set, data_dim)
    LSTM.fit(features_set, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
    y_pred = LSTM.predict(test_features)

  
if __name__ == '__main__':
    t = LoadIt2404()
    mains(t)

    # t=LoadIt0604(2)
    # mains(t)
