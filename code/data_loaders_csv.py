
from feature_extract import *
from numpy  import genfromtxt
import logging
import pickle
from sklearn.utils.estimator_checks import check_complex_data
def load_data(normal_stat=False):
    if normal_stat:
        filepath = "C:\\Python3\\data\\0703\\sensor7.csv"
    else:
        filepath = "C:\\Python3\\data\\0703\\sensor8.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data

def load_data0204Y(normal_stat=False):
    if normal_stat:
       filepath = "C:\\Python3\\data\\02-04-2020\\YTrain.csv"
    else:
       filepath = "C:\\Python3\\data\\02-04-2020\\YTest.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data
def load_data0204X(normal_stat=False):
    if normal_stat:
        filepath = "C:\\Python3\\data\\02-04-2020\\XTrain.csv"
    else:
        filepath = "C:\\Python3\\data\\02-04-2020\\XTest.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data
def load_data_3103(normal_stat=False):
    if normal_stat:
        filepath = "C:\\Python3\\data\\3103\\x.csv"
    else:
        filepath = "C:\\Python3\\data\\3103\\y.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data
    
log = "output_featureextract.log"
logging.basicConfig(filename=log,level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
if __name__ == '__main__':
    data = load_data(True)
    
    #actualData = np.reshape(data,(data.shape[0],-1))
    #print(data[0:4])
    #print (gen_fea_custom(data, 1, 9))
    #output = customExtract0703(data,1,10)
    #np.set_printoptions(threshold=np.inf)
  
    train_x = customFeatureExtract1903(data,500)
    train_y = [1,2]
    
    #np.savetxt("C:\\Python3\\data\\array.txt", output, fmt="%s")
    #print(output)
    #print(output.shape,10)     
    #outputlocation = open('C:\\Python3\\data\\0703\\data.p', 'wb')
    #pickle.dump(output, outputlocation)
    #output.close()
    #printdata()
    #print(actualData.shape)
    #x= gen_fea_custom(actualData, 1, 10)
    #np.savetxt("array.txt", x, fmt="%s")
    #customExtract(actualData)
    #print(x)
    
def LoadIt(normal_state = False):
    data = load_data(False)
    rawTrain_X=data[:10000,:] 
    rawTest_X=data[9500:10000,:] 
      
    train_x = customFeatureExtract1903(rawTrain_X,500)
    test_x = customFeatureExtract1903(rawTest_X,500)
    
    train_y = np.arange(train_x.shape[0])
    test_y = np.arange(test_x.shape[0])
    #test_y = np.arange(9500,9500+test_x.shape[0])
    print(test_y)
    #print(train_y)
    
    
    inputFul = [train_x,train_y,test_x,test_y]
    return inputFul
    
    
    
def LoadIt3103(normal_state = False):
    X_data = load_data_3103(True)
    Y_data = load_data_3103(False)

    rawTrain_X=X_data
    rawTest_X=X_data[:,[0,4,5,10]]
      
      
    train_y = Y_data
    #test_y = Y_data[0,4,5,10]
    test_y = [ Y_data[index] for index in [0,4,5,10] ]
    train_x = customFeatureExtract3103(rawTrain_X,500)
    test_x = customFeatureExtract3103(rawTest_X,500)
    
    
    #test_y = np.arange(9500,9500+test_x.shape[0])
    #print(test_y)
    #print(train_y)
    
    
    inputFul = [train_x,train_y,test_x,test_y]
    return inputFul
def LoadIt0204(normal_state = False):
    
    X_train = load_data0204X(True)
    X_test = load_data0204X(False)
    Y_train = load_data0204Y(True)
    Y_test = load_data0204Y(False)
    
    inputFul = [X_train,X_test,Y_train,Y_test]
    print(inputFul)
    return inputFul
def load_data0304Y(normal_stat=False):
    if normal_stat:
       filepath = "C:\\Python3\\data\\04-03-2020\\ytrain.csv"
    else:
       filepath = "C:\\Python3\\data\\04-03-2020\\ytest.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data
def load_data0304X(normal_stat=False):
    if normal_stat:
        filepath = "C:\\Python3\\data\\04-03-2020\\xtrain.csv"
    else:
        filepath = "C:\\Python3\\data\\04-03-2020\\xtest.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data
def load_data0304X1(normal_stat=False):
    if normal_stat:
        filepath = "C:\\Python3\\data\\04-03-2020\\xtrain.csv"
    else:
        filepath = "C:\\Python3\\data\\04-03-2020\\xtest1.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data
def LoadIt0304Test0(normal_state = False):
    
    X_train = load_data0304X(True)
    X_test = load_data0304X(False)
    Y_train = load_data0304Y(True)
    #Y_test = load_data0204Y(False)
    
    inputFul = [X_train,X_test,Y_train]
    print(inputFul)
    return inputFul
def LoadIt0304Test1(normal_state = False):
    
    X_train = load_data0304X1(True)
    X_test = load_data0304X1(False)
    Y_train = load_data0304Y(True)
    #Y_test = load_data0204Y(False)
    
    inputFul = [X_train,X_test,Y_train]
    print(inputFul)
    return inputFul

def load_data0404(normal_stat=False):
    if normal_stat:
        filepath = "C:\\Python3\\data\\04-04-2020\\X_train4.csv"
    else:
        filepath = "C:\\Python3\\data\\04-04-2020\\Y_train.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data

def load_dataX0404(normal_stat=True):
    if normal_stat:
        filepath = "C:\\Python3\\data\\04-04-2020\\X_test1.csv"
    else:
        filepath = "C:\\Python3\\data\\04-04-2020\\Y_train.csv"
   
    my_data = genfromtxt(filepath, delimiter=',') 
    #with open(filepath, 'r') as file:
    #    reader = csv.reader(file)
    return my_data

def LoadIt0404Test0(normal_state = False):
    
    X_rawtrain = load_data0404(True)
    X_rawtest = X_rawtrain[:,1:3]
    Y_train = load_data0404(False)
    #Y_test = load_data0204Y(False)
    X_train = customFeatureExtract0404(X_rawtrain,5000)
    X_test = customFeatureExtract0404(X_rawtest,5000)
    
    inputFul = [X_train,X_test,Y_train]
    #print(inputFul)
    return inputFul
def LoadIt0404Test1(normal_state = False):
    
    X_rawtrain = load_data0404(True)
    X_rawtest =  load_dataX0404(True)
    Y_train = load_data0404(False)
    #Y_test = load_data0204Y(False)4
    X_train = customFeatureExtract0404(X_rawtrain,5000)
    X_test = customFeatureExtract0404(X_rawtest,5000)
    
    inputFul = [X_train,X_test,Y_train]
    print(inputFul)
    return inputFul
    
    