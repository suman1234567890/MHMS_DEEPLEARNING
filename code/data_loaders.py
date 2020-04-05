import numpy as np
import pickle
import scipy.io as sio
from pickletocsv import convert

def load_data(normal_stat=False):
   
    if normal_stat:
        filepath = "C:\\Python3\\data\\data_normal.p"
    else:
        filepath = "C:\\Python3\\data\\data_seq.p" 
    file = open(filepath, "rb") 
    
    x = pickle.load(file, encoding="bytes")
    return [x[0], x[1], x[2], x[3]]  # retrun train_x, train_y, test_x, test_y

    

if __name__ == '__main__':
    t = load_data(True)
    print (t[0].shape)
    print (t[1].shape )
    print (t[2].shape )
    print (t[3].shape )
   
    
 
    #filepath = "C:\\Python3\\data\\data_seq.p"
    #filepathcsv = "C:\\Python3\\data\\csvfile_Seq.csv"
    #convert(filepath,filepathcsv)
    
    
