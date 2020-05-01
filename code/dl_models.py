from numpy.random import seed
seed(10132017)
import tensorflow 
tensorflow.random.set_seed(18071991)
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv1D,MaxPooling1D
from dbn import SupervisedDBNRegression

dropout_rate = 0.2
FINAL_DIM = 900 
def build_BILSTM(timesteps, data_dim, hidDim=[100,140]):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidDim[0], return_sequences=True, activation='linear'),
                        input_shape=(timesteps, data_dim)))
    model.add(Bidirectional(LSTM(hidDim[1], activation='linear')))
    model.add(Dropout(dropout_rate))
    model.add(Dense(FINAL_DIM,activation='linear'))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model

def build_LSTM1(timesteps, data_dim, hidDim=[100,140]):
    model = Sequential()
    model.add(LSTM(hidDim[0], return_sequences=True,
                        input_shape=(timesteps, data_dim), activation='linear'))
    model.add(LSTM(hidDim[1], activation='linear'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(FINAL_DIM,activation='linear'))
    model.add(Dense(1))
    model.compile(loss="mae", optimizer="sgd")
    return model
def build_LSTM(features_set, data_dim, hidDim=[100,140]):
    
    #model.compile(loss="mae", optimizer="sgd")

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')


    
    return model

def build_CNN(timesteps, data_dim, hidDim=[100,140]):
    model = Sequential()
    model.add(Conv1D(hidDim[0], 3, activation='linear', input_shape=(timesteps, data_dim)))
    model.add(Conv1D(hidDim[1], 3, activation='linear'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(FINAL_DIM,activation='linear'))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model

def build_pre_normalAE(data_dim, X_train, epoch_pretrain=25, hidDim=[100,140]):
    ae = Sequential()
    ae.add(Dense(hidDim[0],activation='linear', input_shape=(data_dim, )))
    ae.add(Dense(hidDim[1], activation='linear'))
    ae.add(Dense(hidDim[0], activation='linear'))
    ae.add(Dense(data_dim, activation='linear'))
    ae.compile(optimizer='rmsprop', loss='mse')
    ae.fit(X_train, X_train, epochs=epoch_pretrain, batch_size=24, shuffle=True, verbose=0)
    model = Sequential()
    model.add(Dense(hidDim[0], input_dim = data_dim, activation='linear'))
    model.add(Dense(hidDim[1], activation='linear'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(FINAL_DIM,activation='linear'))
    model.add(Dense(1))
    model.layers[0].set_weights(ae.layers[0].get_weights())
    model.layers[1].set_weights(ae.layers[1].get_weights())
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


def build_pre_denoiseAE(data_dim, X_train, epoch_pretrain=25,  hidDim=[100,140]):
    ae = Sequential()
    ae.add(Dropout(0.01, input_shape=(data_dim,)))
    ae.add(Dense(hidDim[0],activation='linear'))
    ae.add(Dense(hidDim[1], activation='linear'))
    ae.add(Dense(hidDim[0], activation='linear'))
    ae.add(Dense(data_dim,activation='linear'))
    ae.compile(optimizer='rmsprop', loss='mse')
    ae.fit(X_train, X_train, epochs=epoch_pretrain, batch_size=24, shuffle=True, verbose=0)
    model = Sequential()
    model.add(Dense(hidDim[0], input_dim = data_dim, activation='linear'))
    model.add(Dense(hidDim[1], activation='linear'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(FINAL_DIM,activation='linear'))
    model.add(Dense(1))
    model.layers[0].set_weights(ae.layers[1].get_weights())
    model.layers[1].set_weights(ae.layers[2].get_weights())
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model




def build_RBM(num_bp, epoch_pretrain=25, batch_size=24, hidDim=[100,140]):
    regressor = SupervisedDBNRegression(hidden_layers_structure=hidDim,
                                    learning_rate_rbm=0.01,
                                    learning_rate=0.01,
                                    n_epochs_rbm=epoch_pretrain,
                                    n_iter_backprop=num_bp,
                                    batch_size=batch_size,
                                    activation_function='relu')
    return regressor
