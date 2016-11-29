import os
import numpy as np
import h5py
import ast
import seaborn 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from MainProcessing import ReportClustering
import novainstrumentation.novainstrumentation
from novainstrumentation.novainstrumentation.code.smooth import smooth




def opentxt(filename, filedir):

	if("SN" in filename):
		for file in os.listdir(filedir):
			if(file == "Attributes.txt"):
				inf = open(filedir + "/" + file, 'r')
				AttDic = ast.literal_eval(inf.read())
				inf.close()

				fs = AttDic["fs"]
				n_clusters = AttDic["clusters"]

				ECG_data = np.loadtxt(filename)

			elif(file == "Noise.txt"):
				Noise = np.loadtxt(filedir + "/" + file)

		return ECG_data, fs, n_clusters

	else:
		HeadDic = read_header(filename)
		fs = HeadDic["sampling rate"]
		channel = HeadDic["channels"][HeadDic["sensor"] == "ECG"] + 1

		ECG_data = np.loadtxt(filename)
		ECG_data = ECG_data[:, channel]

		for file in os.listdir(filedir):
			if(file == "Noise.txt"):
				Noise = np.loadtxt(filedir + "/" + file)
			elif(file == "Attributes.txt"):
				inf = open(filedir + "/" + file, 'r')
				AttDic = ast.literal_eval(inf.read())
				inf.close()

				fs = AttDic["fs"]
				n_clusters = AttDic["clusters"]

		return ECG_data, fs, Noise, n_clusters

def read_header(source_file, print_header = False):

	f = open(source_file, 'r')

	f_head = [f.readline() for i in range(3)]

	#convert to diccionary (considering 1 device only)
	head_dic = ast.literal_eval(f_head[1][24:-2])

	return head_dic

file = r'C:\Users\Sara Pestana\Desktop\IWANTTOFINDNOISE\Test\opensignals_000780B383A6_2016-11-16_14-10-33.txt'

signal = np.loadtxt(file)

print("PLOTTING SIGNAL")
plt.plot(signal[:, 3])



fs = 500
n_clusters = 2

ReportClustering(signal[:, 3], fs, n_clusters)


plt.show()


