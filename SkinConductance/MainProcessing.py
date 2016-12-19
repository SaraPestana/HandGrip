import os
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas
import itertools
import matplotlib.patheffects as pte
import matplotlib.gridspec as grid
from scipy.stats import signaltonoise, entropy
from scipy.signal import decimate
from matplotlib.font_manager import FontProperties
from scipy.spatial import distance
from novainstrumentation.smooth import smooth
from IWANTTOFINDNOISE.FeaturesANDClustering.RemoveUnwantedPoints import RemoveUglyDots
from IWANTTOFINDNOISE.FeaturesANDClustering.WindowFeature import WindowStat, findPeakDistance
from IWANTTOFINDNOISE.FeaturesANDClustering.FrequencyFeature import SpectralComponents
from IWANTTOFINDNOISE.FeaturesANDClustering.MultiDClustering import MultiDimensionalClusteringKmeans, MultiDimensionalClusteringAGG
from IWANTTOFINDNOISE.PerformanceMetric.SensEspAcc import GetResults
from IWANTTOFINDNOISE.GenerateThings.PlotSaver import plotClusters, plotDistanceMetric, plotLinearData
from IWANTTOFINDNOISE.GenerateThings.PDFATextGen import get_file_name, pdf_report_creator, pdf_text_closer
from IWANTTOFINDNOISE.GenerateThings.TextSaver import SaveReport
from novainstrumentation.PanThomkinsTest import detect_panthomkins_peaks
from novainstrumentation import peaks
from novainstrumentation.peakdelta import peakdelta
import itertools
from itertools import *

'''
------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------MAIN script------------------------------------------------------
In this script, the main code is developed. This script will open a signal and try
to extract features from it.
The feature extraction process regards applying very simple operations to the
signal in order to find patterns on the signal. The main operators are moving
windows with operations like sum, standard deviation, amplitude variation, etc...
another operator applied to the signal is the frequency spectrum.
After the feature extraction, the feature matrix is created and a clustering
algorithm can be executed in order to classify the signal in its different parts,
namely noisy pattern or non-noisy pattern.
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
'''
#
def ReportClustering(Signal, fs, clusters, win = 50):


    #MainDir = "IWANTOFINDNOISE/TestSignals"
    # SignalKind = os.listdir(MainDir)
    #
    # for SignalFolder in SignalKind:
    #
    # 	SignalDir = MainDir + '/' + SignalFolder
    # 	SignalData = os.listdir(SignalDir)
    #
    # 	for signal in SignalData:
    #
    # 		if ".txt" in signal:
    #

    #----------------------------------------------------------------------------------------------------------------------
    #                            Open Signal (and Pre-Process Data ???)
    #----------------------------------------------------------------------------------------------------------------------

    signal = Signal
    time = np.linspace(0, len(signal)/fs, len(signal))
    print(time)
    osignal = Signal
    signal = signal - np.mean(signal)
    signal = signal/max(abs(signal))

    #----------------------------------------------------------------------------------------------------------------------
    #                                        Extract Features
    #----------------------------------------------------------------------------------------------------------------------

    print("Extracting features...")
    #1 - Std Window
    signalSTD = WindowStat(signal, fs=fs, statTool='std', window_len=(win*fs)/100)

    print("...feature 1 - STD")

    #2 - ZCR
    signalZCR64 = WindowStat(signal, fs=fs, statTool='zcr', window_len=(win*fs)/100)

    print("...feature 2 - ZCR")
    #3 - Sum
    signalSum64 = WindowStat(signal, fs=fs, statTool='sum', window_len=(win*fs)/100)
    signalSum128 = WindowStat(signal, fs=fs, statTool='sum', window_len=(win*fs)/100)

    print("...feature 3 - Sum")
    #4 - Number of Peaks above STD
    signalPKS = WindowStat(signal, fs=fs, statTool='findPks', window_len=(win*fs)/100)
    #signalPKS2 = WindowStat(signal, fs=fs, statTool='findPks', window_len=(win * fs) / 100)

    print("...feature 4 - Pks")
    #5 - Amplitude Difference between successive PKS
    signalADF32 = WindowStat(signal, fs=fs, statTool='AmpDiff', window_len=(win*fs)/100)
    signalADF128 = WindowStat(signal, fs=fs, statTool='AmpDiff', window_len=(2*win*fs)/100)

    print("...feature 5 - AmpDif")
    #6 - Medium Frequency feature
    signalMF = WindowStat(signal, fs=fs, statTool='MF', window_len=(win*fs)/100)
    print("...feature 6 - MF")

    #7 - Frequency Spectrum over time
    Pxx, freqs, bins, im = SpectralComponents(osignal, fs, NFFT=129, show=False) #miss axes

    #Interp - to have the same number of points as the original signal
    signalPxx = np.interp(np.linspace(0, len(Pxx), len(signal)), np.linspace(0, len(Pxx), len(Pxx)), Pxx)

    #8 - Find Peak Distance
    dpks = findPeakDistance(signal, 2*np.std(signal), 0)

    #9 - Smooth signal
    smSignal = smooth(abs(signal), window_len=int(fs/2))
    smSignal = smSignal/max(smSignal)

    smSignal2 = smooth(abs(signal), window_len=int(fs))
    smSignal2 = smSignal2/ max(smSignal2)


    FeatureNamesG = ["Smooth", "Standard Deviation", "MF", "Peaks"]
    FeatureMatrixG = np.array([smSignal, signalSTD, signalMF, signalPKS]).transpose()

    plotLinearData(time, FeatureMatrixG, signal, FeatureNamesG)

    print("Starting Clustering")


    X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(FeatureMatrixG, n_clusters=clusters, Linkage = 'ward', Affinity = 'euclidean')
    #X, y_pred, XPCA, params = MultiDimensionalClusteringKmeans(FeatureMatrix, n_clusters=clusters)


    #----------------------------------------------------------------------------------------------------------------------
    #                  Create Classification Array (1- Noise) (0 - Non-Noise)
    #----------------------------------------------------------------------------------------------------------------------
    #find signal indexes - in this case i will assume that the signal is the majority of the signal


    #print("Derivative")
    #derivada = np.diff(smooth(osignal, window_len=100))
    #derivada = np.array(derivada) / max(abs(derivada))
    #print(derivada)
    #plt.figure("derivada")
    #plt.plot(time[1:len(time)], derivada, color='r')
    #plt.grid(b=True)


    print("Creating Predicted Array...")
    Indexiser = []
    for i in range(0, clusters):
        s = len(y_pred[np.where(y_pred == i)[0]].tolist())
        #s = np.std(signal[np.where(y_pred == i)[0]])
        Indexiser.append(s)


    SigIndex = Indexiser.index(max(Indexiser))
    Prediction = np.ones(np.size(y_pred.tolist()))
    Prediction[np.where(y_pred == SigIndex)[0]] = 1
    print(SigIndex)
    print(Prediction)


    print("Segments of the Signal")
    print("HELLO", SigIndex)
    sig_index = []
    array_segment = []

    for i in range(0, len(y_pred)):
        if y_pred[i] == SigIndex:
            array_segment.append(i)
            sig_index.append(array_segment)
            '''
        elif len(array_segment) > SigIndex:
                sig_index.append(array_segment)
                array_segment = []
        if i == len(y_pred)-1:
            sig_index.append(array_segment)
            array_segment = []
            '''


    print("Plotting Segmentation Signal")
    sig = []
    t = 10 # number of the points
    fs = 500

    for c in range(0, len(sig_index)):
        sig = sig_index[c]

        #print("Signal END")
        signal_ff = smooth(osignal[sig] / max(osignal), window_len=100)



        #print('Peak Amplitude')
        peak_amplitude = peakdelta(signal_ff, 0.02)
        #plt.plot(peak_amplitude[0][:, 0], 'ro', ms = 7)


        indexes_max = peak_amplitude[0][:, 0]
        indexes_min = peak_amplitude[1][:, 0]

        values_min = []
        indexes_peak = []

        for d in range(1, len(indexes_max)):
            indexes_peak = indexes_max[d]
            plt.plot(indexes_max[1:], signal_ff[indexes_max[1:]], 'bo', ms=7, label="Peak Amplitude")
            plt.vlines(x=indexes_peak, ymin=0.6, ymax=1, color='r')
            #plt.annotate("Peak Amplitude", fontsize = 10, xy = (indexes_peak, signal_ff[indexes_peak]))


        # for f in range(0, len(indexes_min)):
        # values_min = indexes_min[f]
        plt.plot(indexes_min, signal_ff[indexes_min], 'go', ms = 7, label="Initial time")

        rise_time = indexes_peak - indexes_min
        values_rise_time = []
        for r in range(0, len(rise_time)):
            values_rise_time = rise_time[r]
            plt.annotate("Rise Time {0}".format(values_rise_time), fontsize=10, xy=(values_rise_time + 3000,  signal_ff[values_rise_time + 5500]))




        plt.plot(signal_ff)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.legend(['Peak_Amplitude'])


        plt.show()












            #print("Initial Time")
            #initial_time = indexes_min
            #print(initial_time)
            #plt.plot(initial_time, signal_ff[initial_time], 'co', ms = 7)
            #plt.annotate("Initial time{0}".format(initial_time), fontsize = 10, xy = (initial_time, signal_ff[initial_time]))


            #indexes_max_r = indexes_max[1:len(indexes_max)]

            #print("Rise time")
            #rise_time = indexes_max_r - indexes_min
            #print(rise_time)


            #for f in range(0, len(indexes_max)):
                #value_max = indexes_max[f]
                #plt.plot(indexes_max, signal_ff[indexes_max], 'bo', ms = 7)
                #plt.vlines(x=indexes_max, ymin=0.6, ymax=1, color='r')

            #plt.plot(rise_time, signal_ff[rise_time], 'bo', lw=3, ms=7)
            #plt.annotate("Rise time{0}".format(rise_time), fontsize=10, xy=(rise_time + 0.01, signal_ff[rise_time] + 0.01))

            #print("Y_max")
            #y_max = signal_ff[indexes_peak]
            #print(y_max)

            #print("Y_min")
            #y_min = signal_ff[initial_time[0]]
            #print(y_min)

            #print("value_y")
            #value_y = (y_max - y_min)*0.5
            #print(half_amplitude)

            #print("Time amplitude")
            #time_amplitude = np.where(signal_ff == value_y)[0][-1]
            #print(time_amplitude)

            #plt.plot(half_amplitude, signal_ff[half_amplitude], 'co', ms=7)
            #plt.annotate("Half amplitude{0}".format(half_amplitude), fontsize=10, xy=(half_amplitude - 0.5, signal_ff[half_amplitude - 0.5]))


















































    #----------------------------------------------------------------------------------------------------------------------
    #                                              Plot Things
    #----------------------------------------------------------------------------------------------------------------------

    print("Plotting...")
    plotClusters(y_pred, signal, time, XPCA, clusters)


    plt.show()



