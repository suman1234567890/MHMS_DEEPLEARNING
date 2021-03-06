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
def mains(t):
	X_train, y_train, X_test = t[0], t[1], t[2]
	data_dim = X_train.shape[1]
	logging.info("X_Train: %s" %(t[0]))
	logging.info("Y_Train: %s" %(t[1]))
	logging.info("X_test: %s" %(t[2]))
	idx_test=0
	tot_iter = 10
	y_predTotal=np.array([])
	for time in range(tot_iter):
		idx_test = idx_test+1
		
		#t = load_data(True)
		
	
		""" 
		linear_regression=build_LN()
		linear_regression.fit(X_train,y_train)
		y_pred = linear_regression.predict(X_test)
		logging.info("   linear predicted result: %s" %(y_pred))
		"""
    	
		
		linear_svr = build_SVR('linear',1000)
		linear_svr.fit(X_train, y_train)
		y_pred = linear_svr.predict(X_test)
		
		y_pred.reshape(1,X_test.shape[0])
		#logging.info(" %f  linear predicted result: %s" %(idx_test, y_pred))

		if (y_predTotal.shape[0] < 1):
			y_predTotal = y_pred
		else:
			y_predTotal = np.append(y_predTotal,y_pred,axis=0)
		#plt.scatter(y_train, y_train, color='blue')
		#plt.show()
	y_predTotal= np.reshape(y_predTotal,(tot_iter,X_test.shape[0]))
	logging.info("  linears predicted mean result: %s" %(y_predTotal.mean(axis=0)))	
	
	
	
		#score= mean_squared_error(y_pred, y_test)
		#mae_score = mean_absolute_error(y_pred, y_test)
		#logging.info("   linear svr result: %f_%f" %(score, mae_score))
	
	
	
	
	
	y_predTotal=np.array([])
	idx_test = 0
	for time in range(tot_iter):
		idx_test = idx_test+1
		NUM_ESTIMATOR = 75		
		NUM_PREEPOCH = 70
		NUM_BPEPOCH = 150
		BATH_SIZE = 100
		rf = build_RF(NUM_ESTIMATOR)
		rf.fit(X_train, y_train)
		y_pred = rf.predict(X_test)
		y_pred.reshape(1,X_test.shape[0])
		#logging.info(" r fores  predict  result: %s" %(y_pred))
		if (y_predTotal.shape[0] < 1):
			y_predTotal = y_pred
		else:
			y_predTotal = np.append(y_predTotal,y_pred,axis=0)
	y_predTotal= np.reshape(y_predTotal,(tot_iter,X_test.shape[0]))
		#plt.scatter(y_train, y_train, color='blue')
		#plt.show()
	
	logging.info("  r fore predicted mean result: %s" %(y_predTotal.mean(axis=0)))	

		#score= mean_squared_error(y_pred, y_test)
		#mae_score = mean_absolute_error(y_pred, y_test)
		#logging.info("   randomforest  result: %f_%f" %(score, mae_score))
		#neural network
	y_predTotal=np.array([])
	idx_test = 0
	for time in range(tot_iter):
		idx_test = idx_test+1
		
		nn_model = build_NN(data_dim)
		nn_model.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = nn_model.predict(X_test)
		
		y_pred.reshape(1,X_test.shape[0])
		#logging.info(" neu  predict  result: %s" %(y_pred))

		if (y_predTotal.shape[0] < 1):
			y_predTotal = y_pred
		else:
			y_predTotal = np.append(y_predTotal,y_pred,axis=0)
	y_predTotal= np.reshape(y_predTotal,(tot_iter,X_test.shape[0]))
	logging.info("  neu predicted mean result: %s" %(y_predTotal.mean(axis=0)))
		#score= mean_squared_error(y_pred, y_test)
		#mae_score = mean_absolute_error(y_pred, y_test)
		#logging.info("   neural network result: %f_%f" %(score, mae_score))
		#AE
	idx_test = 0
	y_predTotal=np.array([])
	for time in range(tot_iter):
		idx_test = idx_test+1
		
		normal_AE = build_pre_normalAE(data_dim, X_train, epoch_pretrain=NUM_PREEPOCH, hidDim=[140,280])
		normal_AE.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = normal_AE.predict(X_test)
		y_pred.reshape(1,X_test.shape[0])
		#logging.info(" ae  predict  result: %s" %(y_pred))

		if (y_predTotal.shape[0] < 1):
			y_predTotal = y_pred
		else:
			y_predTotal = np.append(y_predTotal,y_pred,axis=0)
	y_predTotal= np.reshape(y_predTotal,(tot_iter,X_test.shape[0]))
	logging.info("  ae predicted mean result: %s" %(y_predTotal.mean(axis=0)))
		#score = mean_squared_error(y_pred, y_test)
		#mae_score = mean_absolute_error(y_pred, y_test)
		#logging.info("   normal ae result: %f_%f" %(score, mae_score))
		#denoise AE
		
	idx_test = 0
	y_predTotal=np.array([])
	for time in range(tot_iter):
		idx_test = idx_test+1
		denois_AE = build_pre_denoiseAE(data_dim, X_train, epoch_pretrain=NUM_PREEPOCH, hidDim=[140,280])
		denois_AE.fit(X_train, y_train, epochs=NUM_BPEPOCH, batch_size=BATH_SIZE)
		y_pred = denois_AE.predict(X_test)
		y_pred.reshape(1,X_test.shape[0])
		#logging.info(" denoiseae  predict  result: %s" %(y_pred))

		if (y_predTotal.shape[0] < 1):
			y_predTotal = y_pred
		else:
			y_predTotal = np.append(y_predTotal,y_pred,axis=0)
	y_predTotal= np.reshape(y_predTotal,(tot_iter,X_test.shape[0]))
	logging.info("  denoiseae predicted mean result: %s" %(y_predTotal.mean(axis=0)))

		#score = mean_squared_error(y_pred, y_test)
		#mae_score = mean_absolute_error(y_pred, y_test)
		#logging.info("   denoise ae result: %f_%f" %(score, mae_score))
		# rbf using sigmoid function, feature should be scaled to -1 and 1
	
	idx_test = 0
	
	for time in range(tot_iter):
		idx_test = idx_test+1
		
		scaler = MinMaxScaler()
		X_train_rbm = scaler.fit_transform(X_train)
		rbm = build_RBM(NUM_BPEPOCH, NUM_PREEPOCH, batch_size=BATH_SIZE)
		rbm.fit(X_train_rbm, y_train)
		X_test_rbm = scaler.transform(X_test)
		y_pred = rbm.predict(X_test_rbm)
		logging.info("  dbn predict  result: %s" %(y_pred))
	
	
		#score = mean_squared_error(y_pred, y_test)
		#mae_score = mean_absolute_error(y_pred, y_test)
		#logging.info("   dbn result: %f_%f" %(score, mae_score))
		#Bi-directional LSTM
		#t = load_data(False)
		"""
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
	
		
		
if __name__ == '__main__':
	t=LoadIt1704(1)
	mains(t)
	
	#t=LoadIt0604(2)
	#mains(t)
	