import os
import numpy as np
import h5py
import ast
import seaborn 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from SkinConductance.MainProcessing import ReportClustering
from novainstrumentation import *
from novainstrumentation.smooth import smooth




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

file = r'C:\Users\Sara Pestana\Desktop\IWANTTOFINDNOISE\Test\opensignals_000780B383A6_2016-12-05_11-42-12.txt'
#analisado o sinal opensignals_000780B383A6_2016-12-05_10-46-48.txt com win = 36 (boa classificação do sinal)
#analisando o sinal openssignals_000780B38A6_2016-12-05_11-48-55.txt com win = 40 ou win = 50 (obtenho uma boa classificação do sinal)
#evento opensignals_00780B38A6_2016-12-05_11-22-50.txt

#voltar a fazer a análise do sinal 2016-12-05_11-42-12 porque nao me deu todos os segmentos do sinal
#voltar a fazer a análise para o signal opensignals_00780B38A6_2016-12-05_11-48-55.txt

signal = np.loadtxt(file)

print("PLOTTING SIGNAL")
plt.plot(smooth(signal[:, 3]))



fs = 500
n_clusters = 2

ReportClustering(signal[:, 3], fs, n_clusters)


plt.show()


