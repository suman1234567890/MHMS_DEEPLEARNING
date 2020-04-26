from numpy.random import seed
from sklearn.linear_model._base import LinearRegression

seed(10132017)

import tensorflow
tensorflow.random.set_seed(18071991)
from keras.models import Sequential
from sklearn.svm import SVR
from keras.layers.core import Flatten, Dense, Dropout, Activation
from sklearn.ensemble import RandomForestRegressor


dropout_rate = 0.1
FINAL_DIM = 1000
def build_SVR(kernel_func='rbf', C_value=1.0):
    return SVR(kernel=kernel_func, C=C_value)

def build_RF(num_estimator):
    return RandomForestRegressor(n_estimators=num_estimator)

def build_NN(data_dim, hidDim=[100,120]):
    model = Sequential()
    model.add(Dense(hidDim[0],activation='linear', input_shape=(data_dim, )))
    model.add(Dense(hidDim[1], activation='linear'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(FINAL_DIM,activation='linear'))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")   
    return model

def build_LN():
    return LinearRegression()