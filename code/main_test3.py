import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from numpy.random import seed
seed(10132017)

import tensorflow
tensorflow.random.set_seed(18071991)
from data_loaders import load_data
from dl_models import build_BILSTM, build_LSTM, build_CNN, build_pre_normalAE, build_pre_denoiseAE, build_RBM
from normal_models import build_SVR, build_RF, build_NN
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import logging
from sklearn.preprocessing import MinMaxScaler
from feature_extract import gen_fea
from data_loaders_csv import *


log = "output.log"
logging.basicConfig(filename=log,level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
if __name__ == '__main__':
	for time in range(1):
		idx_test = 1
		logging.info("conducting exp.")
		#t = load_data(True)
		t=LoadIt0204()
		#print("T0-")
		#print(t[0])
		#print("T1-")
		#print(t[1])
		linear_svr = build_SVR('linear')
		linear_svr.fit(t[0], t[2])
		y_pred = linear_svr.predict(t[1])
		logging.info("   linear predicted result: %s" %(y_pred))
		#score= mean_squared_error(y_pred, t[3])
		#mae_score = mean_absolute_error(y_pred, t[3])
		#logging.info("   linear svr result: %f_%f" %(score, mae_score))
		
		rbf_svr = build_SVR('rbf')
		rbf_svr.fit(t[0], t[2])
		y_pred = rbf_svr.predict(t[1])
		logging.info("   RBR predicted result: %s" %(y_pred))
		#score= mean_squared_error(y_pred, t[1])
		#mae_score = mean_absolute_error(y_pred, t[1])
		#logging.info("   rbf svr result: %f_%f" %(score, mae_score))
    	
    	
    	#NUM_ESTIMATOR = 50		
		#NUM_PREEPOCH = 30
		#NUM_BPEPOCH = 30
		#BATH_SIZE = 24
		rf = build_RF(50)
		rf.fit(t[0], t[2])
		y_pred = rf.predict(t[1])
		#score= mean_squared_error(y_pred, y_test)
		#mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   randomforest  result: %s" %(y_pred))
		
		nn_model = build_NN(t[0].shape[1])
		nn_model.fit(t[0], t[2], epochs=30, batch_size=24)
		y_pred = nn_model.predict(t[1])
		logging.info("   neural network result: %s" %(y_pred))
		
		data_dim=t[0].shape[1]
		
    	#normal_AE = build_pre_normalAE(data_dim,t[0])
    	normal_AE = build_pre_normalAE(data_dim, t[0], epoch_pretrain=20, hidDim=[140,280])
		
		normal_AE.fit(t[0],t[2])
		y_pred = normal_AE.predict(t[1])
		
		
		"""
		
		X_train, y_train, X_test, y_test = t[0], t[1], t[2], t[3]
		data_dim = X_train.shape[1]
		logging.info("X_Train: %s" %(t[0]))
		logging.info("Y_Train: %s" %(t[1]))
		logging.info("X_test: %s" %(t[2]))
		logging.info("y_test: %s" %(t[3]))
    	#liner svm
		linear_svr = build_SVR('linear')
		linear_svr.fit(X_train, y_train)
		y_pred = linear_svr.predict(X_test)
		logging.info("   linear predicted result: %s" %(y_pred))
		score= mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   linear svr result: %f_%f" %(score, mae_score))
		rbf_svr = build_SVR('rbf')
		rbf_svr.fit(X_train, y_train)
		y_pred = rbf_svr.predict(X_test)
		score= mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   rbf svr result: %f_%f" %(score, mae_score))
    	#random forst
		NUM_ESTIMATOR = 50		
		NUM_PREEPOCH = 30
		NUM_BPEPOCH = 30
		BATH_SIZE = 24
		rf = build_RF(NUM_ESTIMATOR)
		rf.fit(X_train, y_train)
		y_pred = rf.predict(X_test)
		score= mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   randomforest  result: %f_%f" %(score, mae_score))
		#neural network
		nn_model = build_NN(data_dim)
		nn_model.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = nn_model.predict(X_test)
		score= mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   neural network result: %f_%f" %(score, mae_score))
		#AE
		normal_AE = build_pre_normalAE(data_dim, X_train, epoch_pretrain=NUM_PREEPOCH, hidDim=[140,280])
		normal_AE.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = normal_AE.predict(X_test)
		score = mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   normal ae result: %f_%f" %(score, mae_score))
		#denoise AE
		denois_AE = build_pre_denoiseAE(data_dim, X_train, epoch_pretrain=NUM_PREEPOCH, hidDim=[140,280])
		denois_AE.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = denois_AE.predict(X_test)
		score = mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   denoise ae result: %f_%f" %(score, mae_score))
		# rbf using sigmoid function, feature should be scaled to -1 and 1
		scaler = MinMaxScaler()
		X_train_rbm = scaler.fit_transform(X_train)
		rbm = build_RBM(NUM_BPEPOCH, NUM_PREEPOCH, batch_size=BATH_SIZE)
		rbm.fit(X_train_rbm, y_train)
		X_test_rbm = scaler.transform(X_test)
		y_pred = rbm.predict(X_test_rbm)
		score = mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   dbn result: %f_%f" %(score, mae_score))
		#Bi-directional LSTM
		#t = load_data(False)
		
		t=LoadIt()
		X_train, y_train, X_test, y_test = t[0], t[1], t[2], t[3]
		data_dim = X_train.shape[2]
		
		timesteps = X_train.shape[1]
		biLSTM = build_BILSTM(timesteps, data_dim)
		biLSTM.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = biLSTM.predict(X_test)
		score = mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   bi-lstm result: %f_%f" %(score, mae_score))
		#LSTM
		LSTM = build_LSTM(timesteps, data_dim)
		LSTM.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = LSTM.predict(X_test)
		score = mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   lstm result: %f_%f" %(score, mae_score))
		#CNN
		CNN = build_CNN(timesteps, data_dim)
		CNN.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = CNN.predict(X_test)
		score = mean_squared_error(y_pred, y_test)
		mae_score = mean_absolute_error(y_pred, y_test)
		logging.info("   cnn result: %f_%f" %(score, mae_score))
		
		"""