import csv
import pickle
import numpy as np


def convert(path_pickle,path_csv):

    x = []
    with open(path_pickle,'rb') as f:
        x = pickle.load(f, encoding="bytes")

    with open(path_csv,'w') as f:
        writer = csv.writer(f)
        for line in x: writer.writerow(line)