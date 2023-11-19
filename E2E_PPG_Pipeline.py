# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:02:05 2022

@author: mofeli
"""
# Import necessary libraries and modules
from PPG_SQA import ppg_sqa
from PPG_Reconstruction import ppg_reconstruction
from Clean_PPG_Extraction import clean_segments_extraction
from PPG_HRV_Extraction import PPG_HRV_Extraction
from ppg_peak_detection import peak_detection
from utils import check_and_resample, bandpass_filter



import warnings
warnings.filterwarnings("ignore")




def e2e_hrv_extraction(ppg, timestamp,  sampling_rate, window_length_sec, reconstruction_model_parameters):
    
    # Check if resampling is needed and perform resampling if necessary
    ppg, sampling_rate = check_and_resample(sig=ppg, fs=sampling_rate)

    # Bandpass filter parameters
    lowcut = 0.5  # Lower cutoff frequency in Hz
    highcut = 3  # Upper cutoff frequency in Hz
    
    # Apply bandpass filter
    ppg_filtered = bandpass_filter(sig=ppg, fs=sampling_rate, lowcut=lowcut, highcut=highcut)

    # Run signal quality assessment
    clean_indices, noisy_indices = ppg_sqa(sig=ppg_filtered, sampling_rate=sampling_rate)
    
    # Run PPG reconstruction 
    ppg_signal, clean_indices, noisy_indices = ppg_reconstruction(ppg_filtered, clean_indices, noisy_indices, sampling_rate, reconstruction_model_parameters[0], reconstruction_model_parameters[1])
    
    # Calculate the window length for HR and HRV extraction in terms of samples
    window_length = window_length_sec*sampling_rate

    # Scan clean parts of the signal and extract clean segments with the specified window length
    clean_segments = clean_segments_extraction(sig=ppg_signal, noisy_indices=noisy_indices, window_length=window_length)
    
    # Check if clean segments are found, if not, print a message
    if len(clean_segments) == 0:
        print('No clean ' + str(window_length_sec) + ' seconds segment was detected in the signal!')
    else:
        # Print the number of clean segments found
        print(str(len(clean_segments)) + ' clean ' + str(window_length_sec) + ' seconds segments was detected in the signal!' )

        # Run PPG Peak detection
        peaks, sampling_rate_new = peak_detection(clean_segments, sampling_rate)
        
        # Perform HRV extraction
        hrv_data = PPG_HRV_Extraction(clean_segments, start_timestamp_segments, peaks, sampling_rate_new, window_length_min)

    print('Done!')
    return hrv_data






