"""
This script define various feature extraction methods
"""
#! /usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import pickle 
from collections import OrderedDict
import csv
import glob
import os
from numpy import mean, sqrt, square
import scipy.stats as sts
from subprocess import Popen, PIPE
from pywt import WaveletPacket
from numpy.random.mtrand import sample


def rms_fea(a):
	return np.sqrt(np.mean(np.square(a)))

def var_fea(a):
	return np.var(a)

def max_fea(a):
	return np.max(np.abs(a))

def pp_fea(a):
	return np.max(a)-np.min(a)

def skew_fea(a):
	return sts.skew(a)

def kurt_fea(a):
	return sts.kurtosis(a)

def wave_fea(a):
	wp = WaveletPacket(a,'db1', maxlevel=8)
	nodes = wp.get_level(8, "freq")
	return np.linalg.norm(np.array([n.data for n in nodes]), 2)

def spectral_kurt(a):
	
	N=a.shape[0]
	mag = np.abs(np.fft.fft(a))
	mag	= mag[1:N//2]*2.00/N
	return sts.kurtosis(mag)

def spectral_skw(a):
	N= a.shape[0]
	mag = np.abs(np.fft.fft(a))
	mag	= mag[1:N//2]*2.00/N
	return sts.skew(mag)

def spectral_pow(a):
	N= a.shape[0]
	mag = np.abs(np.fft.fft(a))
	mag	= mag[1:N//2]*2.00/N
	return np.mean(np.power(mag, 3))


def extract_fea(data, num_stat = 10):
	# input: time_len * dim_fea  -> dim_fea*9
	data_fea = []
	dim_feature = 1
	for i in range(dim_feature):
		data_slice = data
		data_fea.append(rms_fea(data_slice))
		data_fea.append(var_fea(data_slice))
		data_fea.append(max_fea(data_slice))
		data_fea.append(pp_fea(data_slice))
		data_fea.append(skew_fea(data_slice))
		data_fea.append(kurt_fea(data_slice))
		data_fea.append(wave_fea(data_slice))
		data_fea.append(spectral_kurt(data_slice))
		data_fea.append(spectral_skw(data_slice))
		data_fea.append(spectral_pow(data_slice))
	data_fea = np.array(data_fea)
	return data_fea.reshape((1,dim_feature*num_stat))

def extract_fea_cust(data_slice, num_stat = 10):
	# input: time_len * dim_fea  -> dim_fea*9
	data_fea = []
	dim_feature = 1
	data_fea.append(rms_fea(data_slice))
	#data_fea.append(rms_fea(data_slice))
	#data_fea.append(var_fea(data_slice))
	data_fea.append(max_fea(data_slice))
	#data_fea.append(pp_fea(data_slice))
	#data_fea.append(skew_fea(data_slice))
	#data_fea.append(kurt_fea(data_slice))
	#data_fea.append(wave_fea(data_slice))
	#data_fea.append(spectral_kurt(data_slice))
	#data_fea.append(spectral_skw(data_slice))
	#data_fea.append(spectral_pow(data_slice))
	data_fea = np.array(data_fea)
	return data_fea.reshape((1,dim_feature*num_stat))
		
			
def gen_fea(data,time_steps = 20,num_stat = 10):
	"""
	input: 
		@data: raw time series data, [data size,  sequence length]
		@time_steps: the number of windows, and it can be one
		@num_stat: number of features for each window
	"""
	data_num = data.shape[0]
	len_seq = data.shape[1]
	window_len = len_seq//time_steps
	if window_len == len_seq:
		new_data = np.ones((data_num,num_stat), dtype=np.float32)
		for idxdata, sig_data in enumerate(data):
			new_data[idxdata,:] = extract_fea(sig_data)
	else:
		new_data = np.ones((data_num,time_steps,num_stat), dtype=np.float32)
		for idxdata, sig_data in enumerate(data):
			for i in range(time_steps):
				start = i*window_len
				end = (i+1)*window_len	
				
				temp_data = extract_fea(sig_data[start:end])
				new_data[idxdata,i,:] = temp_data
	return new_data

def gen_fea_custom(data,time_steps = 20,num_stat = 10):
	"""
	input: 
		@data: raw time series data, [data size,  sequence length]
		@time_steps: the number of windows, and it can be one
		@num_stat: number of features for each window
	"""
	data_num = data.shape[0]
	
	
	new_data = np.ones((data_num,num_stat), dtype=np.float32)
	for idxdata, sig_data in enumerate(data):
		
		new_data[idxdata,:] = extract_fea(sig_data)
	
	return new_data
def printdata():
	mag	= mag[1:N/2]*2.00/N
if __name__ == '__main__':
	t = load_data(False)
	gen_fea(data, time_steps, num_stat)
	
def customExtract(data_slice):
	print("RMS")
	print(rms_fea(data_slice))
	print("VAR")
	print(var_fea(data_slice))
	print("MAX")
	print(max_fea(data_slice))
	print("PP")
	print(pp_fea(data_slice))
	print("SKEW")
	print(skew_fea(data_slice))
	print("KURT")
	print(kurt_fea(data_slice))
	#data_fea.append(wave_fea(data_slice))
	print("Spectral_KURT")
	print(spectral_kurt(data_slice))
	print("Spectral_SKW")
	print(spectral_skw(data_slice))
	print("Spectral_POW")
	print(spectral_pow(data_slice))

def customExtract0703(data,time_steps = 1,num_stat=10):
	data_num = data.shape[0]
	print(data.shape)
	"""
	window_len = time_steps
	new_data = np.ones((data_num,time_steps,num_stat), dtype=np.float32)
	for idxdata, sig_data in enumerate(data):
		print(idxdata)
		new_data[idxdata,:] = extract_fea(sig_data)
	"""
	
	return gen_fea(data,time_steps,num_stat)


def customFeatureExtract1903(data,sample_size):
	#print(data)
	#new_data = np.ones((data.shape[0],data.shape[1]), dtype=np.float32)
	row = 0
	column = 0
	no_of_feature = 10
	result_array_final = np.array([])
	while (row < data.shape[0]):
		result_array = np.array([])
		column = 0
		new_data = data[row:row+sample_size]
		while (column < data.shape[1]):
			#print("input data")
			#print(new_data[:,column])
			y = extract_fea_cust(new_data[:,column])
			z = np.reshape(y, ( 1,no_of_feature))
			#print("extracted feature Z")
			print(z)
			#print("extracted feature Y")
			#print(y)
			#print("extracted feature")
			if (result_array.shape[0] < 1):
				result_array = z
			else:
				result_array = np.append(result_array, z,axis=1)
			column = column + 1
					
		row = row + sample_size
		if (result_array_final.shape[0] < 1):
			result_array_final = result_array
		else:
			result_array_final = np.append(result_array_final,result_array,axis=0)
		print("Row" ,row)
		print(result_array)
		
	print("Final",result_array_final.shape)
	print(result_array_final)	
	return result_array_final
def customFeatureExtract3103(data,sample_size):
	#print(data)
	#new_data = np.ones((data.shape[0],data.shape[1]), dtype=np.float32)
	row = 0
	column = 0
	no_of_feature = 10
	result_array_final = np.array([])
	while (column < data.shape[1]):
		result_array = np.array([])
		
		#print("input data")
		#print(new_data[:,column])
		print("----------")
		print("column" )
		print(column)
		print(data[:,column])
		print("----------")
		y = extract_fea_cust(data[:,column])
		result_array = np.reshape(y, ( 1,no_of_feature))
		#print("extracted feature Z")
		#print(result_array)
		#print("extracted feature Y")
		#print(y)
		#print("extracted feature")
		
		column = column + 1
					
		
		if (result_array_final.shape[0] < 1):
			result_array_final = result_array
		else:
			result_array_final = np.append(result_array_final,result_array,axis=0)
		#print("Row" ,column)
		#print(result_array)
		
	#print("Final",result_array_final.shape)
	#print(result_array_final)	
	return result_array_final

def customFeatureExtract0404(data,sample_size):
	#print(data)
	#new_data = np.ones((data.shape[0],data.shape[1]), dtype=np.float32)
	row = 0
	column = 0
	no_of_feature = 2
	result_array_final = np.array([])
	while (column < data.shape[1]):
		result_array = np.array([])
		
		#print("input data")
		#print(new_data[:,column])
		print("----------")
		print("column" )
		print(column)
		print(data[:,column])
		print("----------")
		y = extract_fea_cust(data[:,column],no_of_feature)
		result_array = np.reshape(y, ( 1,no_of_feature))
		#print("extracted feature Z")
		#print(result_array)
		#print("extracted feature Y")
		#print(y)
		#print("extracted feature")
		
		column = column + 1
					
		
		if (result_array_final.shape[0] < 1):
			result_array_final = result_array
		else:
			result_array_final = np.append(result_array_final,result_array,axis=0)
		#print("Row" ,column)
		#print(result_array)
		
	#print("Final",result_array_final.shape)
	#print(result_array_final)	
	return result_array_final
