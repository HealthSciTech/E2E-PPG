# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:02:05 2022

@author: mofeli
"""
# Import necessary libraries and modules
from PPG_SQA import PPG_SQA
from PPG_Reconstruction import ppg_reconstruction
from Clean_PPG_Extraction import clean_ppg_extraction
from PPG_HRV_Extraction import PPG_HRV_Extraction
import pandas as pd
from scipy.signal import resample
from PPG_Peak_Detection import peak_detection
import heartpy as hp
import numpy as np
import os
import glob

import warnings
warnings.filterwarnings("ignore")



# Specify the input file and sample rate
file = '201902020222_Data.csv'
sample_rate = 20

try:
    # Attempt to load data from the specified CSV file
    input_data = pd.read_csv(file, delim_whitespace=True, usecols=['timestamp','ppg'])
    ppg = input_data['ppg'].to_numpy()
    timestamp = input_data['timestamp'].to_numpy()
except:
    print('The input file can not be read, check the data type please!')
    

print('Pipeline started for ' + file)

# Check if resampling is needed and perform resampling if necessary
if sample_rate != 20:
    resampling_rate = 20/sample_rate
    ppg = resample(ppg, int(len(ppg)*resampling_rate))
    sample_rate = 20

               
# Filtering the PPG signal using bandpass filter
cutoff =[0.5,3]
ppg_filtered = hp.filtering.filter_signal(ppg, sample_rate=sample_rate, cutoff=cutoff, filtertype='bandpass')

# Signal quality assessment
x_reliable, gaps = PPG_SQA(ppg_filtered, sample_rate, doPlot=False)

# PPG reconstruction for noises less than 15 sec using GAN model
execfile('GAN.py')
ppg_signal, x_reliable, gaps = ppg_reconstruction(G, device ,ppg_filtered, x_reliable, gaps, sample_rate)


# Set the window length for HR and HRV extraction
window_length_min = 1.5
# Extract clean PPG segments for the specified window length
clean_segments, start_timestamp_segments = clean_ppg_extraction(ppg_signal, gaps, window_length_min, sample_rate, timestamp)

# Check if clean segments are found, if not, print a message
if len(clean_segments) == 0:
    print('No clean ' + str(window_length_min) + ' min ppg was detected in signal!')
else:
    # Print the number of clean segments found
    print(str(len(clean_segments)) + ' clean ' + str(window_length_min) + ' min ppg was detected in signal' )
    
    ## PPg Peak detection
    peaks = peak_detection(clean_segments, sample_rate)
    # Perform HRV extraction
    hrv_data = PPG_HRV_Extraction(clean_segments, start_timestamp_segments, peaks, sample_rate, window_length_min)
    
    # Save HRV data to a CSV file
    filename = 'HRV_' + file
    hrv_data.to_csv(filename, index=False)


print('Done!')






