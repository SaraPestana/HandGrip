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
from novainstrumentation.novainstrumentation.code.smooth import smooth
from IWANTTOFINDNOISE.FeaturesANDClustering.RemoveUnwantedPoints import RemoveUglyDots
from IWANTTOFINDNOISE.FeaturesANDClustering.WindowFeature import WindowStat, findPeakDistance
from IWANTTOFINDNOISE.FeaturesANDClustering.FrequencyFeature import SpectralComponents
from IWANTTOFINDNOISE.FeaturesANDClustering.MultiDClustering import MultiDimensionalClusteringKmeans, MultiDimensionalClusteringAGG
from IWANTTOFINDNOISE.PerformanceMetric.SensEspAcc import GetResults
from IWANTTOFINDNOISE.GenerateThings.PlotSaver import plotClusters, plotDistanceMetric, plotLinearData
from IWANTTOFINDNOISE.GenerateThings.PDFATextGen import get_file_name, pdf_report_creator, pdf_text_closer
from IWANTTOFINDNOISE.GenerateThings.TextSaver import SaveReport
from novainstrumentation.novainstrumentation.code.tests.Peak_find import detect_peaks
from novainstrumentation.novainstrumentation.code import peaks
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
def ReportClustering(Signal, fs, clusters, win=50):


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


    FeatureNamesG = ["Smooth", "Sum", "Standard Deviation"]
    FeatureMatrixG = np.array([smSignal, signalSum128, signalSTD]).transpose()

    plotLinearData(time, FeatureMatrixG, signal, FeatureNamesG)

    print("Starting Clustering")


    X, y_pred, XPCA, params = MultiDimensionalClusteringAGG(FeatureMatrixG, n_clusters=clusters, Linkage = 'ward', Affinity = 'euclidean')
    #X, y_pred, XPCA, params = MultiDimensionalClusteringKmeans(FeatureMatrix, n_clusters=clusters)


    #----------------------------------------------------------------------------------------------------------------------
    #                  Create Classification Array (1- Noise) (0 - Non-Noise)
    #----------------------------------------------------------------------------------------------------------------------
    #find signal indexes - in this case i will assume that the signal is the majority of the signal


    print("Derivative")
    derivada = np.diff(smooth(osignal, window_len=100))
    derivada = np.array(derivada) / max(abs(derivada))
    print(derivada)
    plt.figure("derivada")
    plt.plot(time[1:len(time)], derivada, color='r')
    plt.grid(b=True)


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


    print("Indexes of the Signal")
    sig_index = []
    lock = False
    a = []
    for i in range(0, len(y_pred)):
        if y_pred[i] == SigIndex:
            a.append(i)
        else:
            if len(a) > SigIndex:
                sig_index.append(a)
                a = []
    print("hello", sig_index)
    print("hello", len(sig_index))

    print("Plotting Segmentation Signal")
    sig = []
    t = 10  # number of the points

    for c in range(0, len(sig_index)):
        sig = sig_index[c]
        print(sig)

        print("Signal END")
        signal_ff = smooth(osignal[sig] / max(osignal), window_len=100)
        plt.figure()
        plt.plot(signal_ff)
        plt.title('Segmentation signal')


        win_width = t * fs
        mean_sig = np.zeros(len(signal_ff) / win_width)
        numstps = int((len(signal_ff) - win_width))  # Number of windows
        for d in range(0, len(signal_ff) - win_width):
            print("S1")
            s1 = np.array(signal_ff[d:d + win_width])
            print(s1)

            print("Maximum value")
            max_value = max(signal_ff[:2800])
            print(max_value)


            print("Max index")
            max_index = np.argmax(signal_ff[:2800])
            print(max_index)

            print("T_derivative")
            t_derivative = [max_index - 1300, max_index]
            print(t_derivative)

            print("Value x")
            x = np.array(derivada[t_derivative[0]:t_derivative[1]])
            print(x)

            print("Final value")
            final_index = max_index - 1300 + np.where(x > 0.1)[0][0]
            print(final_index)
            initial_time = 10

            print("Latency time")
            latency_time = final_index - initial_time
            plt.plot(final_index, signal_ff[final_index], 'co', lw=3, ms=7)
            plt.annotate("Latency time {0}".format(latency_time), fontsize = 10, xy = (final_index + 0.02, signal_ff[final_index] + 0.02))

            print("Initial time")

            print(initial_time)
            #plt.plot(time[initial_time], signal_ff[initial_time] / max(osignal), 'r*', lw=3, ms=10)
            #plt.annotate("Initial time", fontsize=10, xy=(time[initial_time], signal_ff[initial_time] / max(osignal)), arrowprops=dict(facecolor='black', shrink=0.5), horizontalalignment='right', verticalalignment='top')

            print("Latency time")

            print(latency_time)
            #plt.annotate("Latency time", fontsize = 12, xy = (t[final_index], s[final_index] / max(s)), xytext = (t[initial_time] - 10.0, s[initial_time] / max(s)), arrowprops=dict(arrowstyle = "->", connectionstyle = 'arc3, rad=0.5', alpha = 2), horizontalalignment = 'left', verticalalignment = 'top')
            #plt.annotate("Latency time", fontsize=10, xy=(time[final_index], signal_ff[final_index] / max(osignal)), arrowprops=dict(facecolor='black', shrink=0.5), horizontalalignment='left', verticalalignment='bottom')


            #print("Amplitude")
            #amplitude = max(signal_ff[:2800]) / max(osignal)
            #print(amplitude)
            print("Maximum value")
            amplitude1 = max(signal_ff[:2800])
            print(amplitude1)
            plt.plot(max_index, amplitude1, 'ro', ms = 7)

            plt.annotate("Maximum value{0}".format(round(amplitude1, 3)), fontsize=10, xy=(max_index + 0.02, amplitude1 + 0.02))

            print("Rise timne")
            rise_time = max_index + 1300 - latency_time
            print(rise_time)
            plt.plot(rise_time, signal_ff[rise_time], 'bo', lw=3, ms=7)
            plt.annotate("Rise time{0}".format(rise_time), fontsize=10, xy=(rise_time + 0.02, signal_ff[rise_time] + 0.02))


            print("Recovery half time")
            half_recovery_time = np.where(signal_ff[:2800] > amplitude1 * 0.5)[0][-1]
            plt.plot(half_recovery_time, signal_ff[half_recovery_time], 'go', ms=7)
            print(half_recovery_time)
            plt.annotate("Recovery half time{0}".format(half_recovery_time), fontsize=10, xy=(half_recovery_time + 0.02, signal_ff[half_recovery_time] + 0.02))

























    #----------------------------------------------------------------------------------------------------------------------
    #                                              Plot Things
    #----------------------------------------------------------------------------------------------------------------------

    print("Plotting...")
    plotClusters(y_pred, signal, time, XPCA, clusters)


    plt.show()



