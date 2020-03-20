
from feature_extract import *
from numpy  import genfromtxt
import logging
import pickle
def load_data(normal_stat=False):
    if normal_stat:
        filepath = "C:\\Python3\\data\\0703\\sensor7.csv"
    else:
        filepath = "C:\\Python3\\data\\Input.csv" 
   
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
    output = customFeatureExtract1903(data,500)
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