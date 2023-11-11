# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:02:05 2022

@author: mofeli
"""
from PPG_SQA import PPG_SQA
from PPG_Reconstruction import ppg_reconstruction
from Clean_PPG_Extraction import clean_ppg_extraction
from PPG_HRV_Extraction import PPG_HRV_Extraction
import pandas as pd
from scipy.signal import resample

import heartpy as hp
import numpy as np
import os
import glob

import warnings
warnings.filterwarnings("ignore")




file = '201902020222_Data.csv'
sample_rate = 20

try:
    # load data
    input_data = pd.read_csv(file, delim_whitespace=True, usecols=['timestamp','ppg'])
    ppg = input_data['ppg'].to_numpy()
    timestamp = input_data['timestamp'].to_numpy()
except:
    print('The input file can not be read, check the data type please!')
    

print('Pipeline started for ' + file)


if sample_rate != 20:
    # Resampling the signal to the desired sample rate
    resampling_rate = 20/sample_rate
    ppg = resample(ppg, int(len(ppg)*resampling_rate))
    sample_rate = 20

               
# filtering
cutoff =[0.5,3]
ppg_filtered = hp.filtering.filter_signal(ppg, sample_rate=sample_rate, cutoff=cutoff, filtertype='bandpass')

# Signal quality assessment
x_reliable, gaps = PPG_SQA(ppg_filtered, sample_rate, doPlot=False)

# PPG reconstruction for noises less than 15 sec
execfile('GAN.py')
ppg_signal, x_reliable, gaps = ppg_reconstruction(G, device ,ppg_filtered, x_reliable, gaps, sample_rate)


# window_length shows the length of the segments that HR and HRV are extracted from them. For example, if you need 5 min HR/HRV, change this parameter to 5. 
window_length_min = 1.5
clean_segments, start_timestamp_segments = clean_ppg_extraction(ppg_signal, gaps, window_length_min, sample_rate, timestamp)


if len(clean_segments) == 0:
    print('No clean ' + str(window_length_min) + ' min ppg was detected in signal!')
else:
    ## PPg Peak detection
    #segmentized_signal, indeces = PPG_Peak(np.asarray(ppg_cleans[0]), sample_rate, fc = cutoff, order = 4,
    #                                       btype = 'bandpass', seconds = 15, overlap = 0, minlen = 15, doplot = False)
    print(str(len(clean_segments)) + ' clean ' + str(window_length_min) + ' min ppg was detected in signal' )
    # HRV indices extraction
    hrv_data = PPG_HRV_Extraction(clean_segments, start_timestamp_segments, sample_rate, window_length_min)
    
    filename = 'HRV_' + file
    hrv_data.to_csv(filename, index=False)


print('Done!')






