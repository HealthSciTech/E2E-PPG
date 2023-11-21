# -*- coding: utf-8 -*-

import neurokit2 as nk
import heartpy as hp
from heartpy.datautils import rolling_mean
import numpy as np
from scipy import signal
from kazemi_peak_detection import ppg_peaks
from ppg_sqa import sqa
from ppg_reconstruction import reconstruction
from ppg_clean_extraction import clean_seg_extraction
from utils import normalize_data, get_data, bandpass_filter, check_and_resample
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


def peak_detection(
        clean_segments: list, 
        sampling_rate: int, 
        method: str ='nk') -> Tuple[list, int]:
    '''
    Detect peaks in clean PPG segments using specified peak detection method.
    
    Input parameters
        clean_segments (list): List of clean PPG segments with the specified window length and their starting index.
        sampling_rate: Sampling rate of the PPG signal.
        method (str): Peak detection method. Valid inputs: 'nk', 'kazemi', and  'heartpy'. The default is 'nk'. (optional)

    Returns
        total_peaks (list): List of lists, each containing the detected peaks for a corresponding clean segment.
        sampling_rate_new (int): Updated sampling rate after any necessary signal processing.

    '''
    # Initialize a list to store total peaks
    total_peaks = []
    
    # Check the deisred peak detection method
    if method == 'nk':
        # Neurokit method
        upsampling_rate = 2
        sampling_rate_new = sampling_rate * upsampling_rate
        
        for i in range(len(clean_segments)):
            # Normalize PPG signal
            ppg_normed = normalize_data(clean_segments[i][1])
            
            # Upsampling the signal
            resampled = signal.resample(ppg_normed, len(ppg_normed) * upsampling_rate)
            
            # Perform peak detection 
            ppg_cleaned = nk.ppg_clean(resampled, sampling_rate=sampling_rate_new)
            info  = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate_new)
            peaks = info["PPG_Peaks"]
            
            # Add peaks of the current segment to the total peaks
            total_peaks.append(peaks)
            
        # Return total peaks and updated sampling rate 
        return total_peaks, sampling_rate_new
    
    elif method == 'kazemi':
        # Kazemi method
        for i in range(len(clean_segments)):
            # Perform peak detection
            peaks, sampling_rate_new = ppg_peaks(np.asarray(clean_segments[i][1]), sampling_rate, seconds = 15, overlap = 0, minlen = 15)
            
            # Add peaks of the current segment to the total peaks
            total_peaks.append(peaks)
            
        # Return total peaks and updated sampling rate 
        return total_peaks, sampling_rate_new

    elif method == 'heartpy':
        # HeartPy method
        for i in range(len(clean_segments)):
            # Perform peak detection
            rol_mean = rolling_mean(clean_segments[i][1], windowsize = 0.75, sample_rate = sampling_rate)
            wd = hp.peakdetection.detect_peaks(np.array(clean_segments[i][1]), rol_mean, ma_perc = 20, sample_rate = sampling_rate)
            peaks = wd['peaklist']
            
            # Add peaks of the current segment to the total peaks
            total_peaks.append(peaks)
            
        # Return total peaks and updated sampling rat
        return total_peaks, sampling_rate
        
    else:
        print("Invalid method. Please choose from 'neurokit', 'kazemi', or 'heartpy'")
        return None




if __name__ == "__main__":
    # Import a sample data
    FILE_NAME = "201902020222_Data.csv"
    SAMPLING_FREQUENCY = 20
    input_sig = get_data(file_name=FILE_NAME)
    
    # Check if resampling is needed and perform resampling if necessary
    input_sig, sampling_rate = check_and_resample(sig=input_sig, fs=SAMPLING_FREQUENCY)
    
    # Bandpass filter parameters
    lowcut = 0.5  # Lower cutoff frequency in Hz
    highcut = 3  # Upper cutoff frequency in Hz
    
    # Apply bandpass filter
    filtered_sig = bandpass_filter(sig=input_sig, fs=sampling_rate, lowcut=lowcut, highcut=highcut)

    # Run PPG signal quality assessment.
    clean_indices, noisy_indices = sqa(sig=filtered_sig, sampling_rate=sampling_rate)
    
    # Run PPG reconstruction
    ppg_signal, clean_indices, noisy_indices = reconstruction(sig=filtered_sig, clean_indices=clean_indices, noisy_indices=noisy_indices, sampling_rate=sampling_rate)
    
    # Define a window length for clean segments extraction (in seconds)
    WINDOW_LENGTH_SEC = 90
    
    # Calculate the window length in terms of samples
    window_length = WINDOW_LENGTH_SEC*sampling_rate
    
    # Scan clean parts of the signal and extract clean segments with the specified window length
    clean_segments = clean_seg_extraction(sig=ppg_signal, noisy_indices=noisy_indices, window_length=window_length)
    
    # Display results
    print("Analysis Results:")
    print("------------------")
    # Check if clean segments are found, if not, print a message
    if len(clean_segments) == 0:
        print('No clean ' + str(WINDOW_LENGTH_SEC) + ' seconds segment was detected in the signal!')
    else:
        # Print the number of detected clean segments
        print(str(len(clean_segments)) + ' clean ' + str(WINDOW_LENGTH_SEC) + ' seconds segments was detected in the signal!' )
        
        # Run PPG Peak detection
        peaks, sampling_rate_new = peak_detection(clean_segments=clean_segments, sampling_rate=sampling_rate)
        print("Number of detected peaks in each segment:")
        for pks in peaks:
            print(len(pks))
    