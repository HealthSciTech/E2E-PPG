# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:02:05 2022

@author: mofeli
"""
# Import necessary libraries and modules
from ppg_sqa import PPG_SQA
from PPG_Reconstruction import ppg_reconstruction
from Clean_PPG_Extraction import clean_ppg_extraction
from PPG_HRV_Extraction import PPG_HRV_Extraction
from scipy.signal import resample
from PPG_Peak_Detection import peak_detection
from PPG_Filtering import PPG_bandpass_filter

import warnings
warnings.filterwarnings("ignore")




def HRV_Extraction(ppg, timestamp,  sample_rate, window_length_min, reconstruction_model_parameters):
    
    # Check if resampling is needed and perform resampling if necessary
    if sample_rate != 20:
        resampling_rate = 20/sample_rate
        ppg = resample(ppg, int(len(ppg)*resampling_rate))
        sample_rate = 20
    

    # Bandpass filter parameters
    lowcut = 0.5  # Lower cutoff frequency in Hz
    highcut = 3  # Upper cutoff frequency in Hz
    
    # Apply bandpass filter
    ppg_filtered = PPG_bandpass_filter(ppg, lowcut, highcut, sample_rate)

    
    # Signal quality assessment
    x_reliable, gaps = PPG_SQA(ppg_filtered, sample_rate, doPlot=False)
    
    # PPG reconstruction for noises less than 15 sec using GAN model
    ppg_signal, x_reliable, gaps = ppg_reconstruction(reconstruction_model_parameters[0], reconstruction_model_parameters[1] ,ppg_filtered, x_reliable, gaps, sample_rate)
    
    
    # Extract clean PPG segments for the specified window length
    clean_segments, start_timestamp_segments = clean_ppg_extraction(ppg_signal, gaps, window_length_min, sample_rate, timestamp)
    
    # Check if clean segments are found, if not, print a message
    if len(clean_segments) == 0:
        print('No clean ' + str(window_length_min) + ' min ppg was detected in signal!')
    else:
        # Print the number of clean segments found
        print(str(len(clean_segments)) + ' clean ' + str(window_length_min) + ' min ppg was detected in signal' )
        
        ## PPG Peak detection
        peaks, sample_rate_new = peak_detection(clean_segments, sample_rate)
        # Perform HRV extraction
        hrv_data = PPG_HRV_Extraction(clean_segments, start_timestamp_segments, peaks, sample_rate_new, window_length_min)
        
    
    print('Done!')
    return hrv_data






