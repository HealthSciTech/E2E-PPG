# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:09:18 2022

@author: mofeli
"""

# Import necessary libraries
import neurokit2 as nk
import pandas as pd
import numpy as np
from PPG_SQA import ppg_sqa
from PPG_Reconstruction import ppg_reconstruction
from Clean_PPG_Extraction import clean_segments_extraction
from ppg_peak_detection import peak_detection
from utils import get_data, bandpass_filter

def HRV_parameters(
        peaks: np.ndarray,
        seg_start_idx: int,
        sampling_rate: int,
        window_length: int):
    """
    Calculate HR and HRV parameters from a list of peak indices.

    Input parameters:
        peaks (np.ndarray): Peak indices for a segment.
        seg_start_idx (int): Starting index of a segemnt.
        sampling_rate (int): Sampling rate of the signal.
        window_length (int): Desired window length for HR and HRV extraction in terms of samples.

    Returns:
        HRV_indices (pandas.DataFrame): DataFrame containing HRV indices.
    """
    
    # Calculate window length in minutes
    window_length_min = (window_length/sampling_rate)/60    
    
    # Calculate heart rate
    heart_rate = (len(peaks)/window_length_min)
    
    # Initialize a list to store HRV indices
    HRV_indices = []
    
    # Compute time-domain indices HRV
    hrv_times = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
    hrv_times = hrv_times[['HRV_MeanNN','HRV_SDNN','HRV_RMSSD','HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN','HRV_TINN','HRV_HTI']]
    HRV_indices.append(hrv_times)

    # Computes frequency-domain indices of HRV
    hrv_freqs = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, show=False, psd_method="welch")
    HRV_indices.append(hrv_freqs)
    
    # Computes nonlinear indices of HRV
    hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=False)
    hrv_nonlinear = hrv_nonlinear[['HRV_SD1','HRV_SD2','HRV_SD1SD2','HRV_S']]
    HRV_indices.append(hrv_nonlinear)

    # Concatenate HRV indices into a DataFrame
    HRV_indices = pd.concat(HRV_indices, axis=1)
    
    # Add segment starting and heart rate to the DataFrame
    HRV_indices.insert(0, 'Start_idx', seg_start_idx)
    HRV_indices.insert(1, 'HR', heart_rate)
    
    # HRV_indices['HR'] = heart_rate
    # HRV_indices['Timestamp'] = timestamp
    
    return HRV_indices

    
#     print('Pulse rate: ' + str(len(peaks)*2))
#     plt.figure(figsize=(12,3))
#     # plt.plot(resampled, label = 'PPG signal')
#     # plt.plot(ppg_cleaned, label = 'Cleaned PPG signal')
#     plt.plot(ppg_cleaned)

#     plt.scatter(peaks,ppg_cleaned[peaks], label = 'Peaks', color='r')
#     plt.legend(fontsize=10)
#     plt.xlabel('Time (s)', fontsize=15)
#     plt.ylabel('Amplitude', fontsize=15)



def hrv_extraction(
        clean_segments: list,
        peaks: list,
        sampling_rate: int,
        window_length: int) -> pd.DataFrame:
    
    """
    Calculate HR and HRV parameters from clean segments peak indices.

    Input parameters:
        clean_segments (list): List of clean PPG segments and their starting index.
        peaks (list): List of lists, each containing the detected peaks for a corresponding clean segment.
        sampling_rate (int): Sampling rate of the PPG signal.
        window_length (int): Desired window length for HR and HRV extraction in terms of samples.

    Returns:
    - hrv_data (pandas.DataFrame): DataFrame containing HRV indices for all clean segments.
    """
    
    # Iterate through the segments to extract HRV parameters
    for i in range(len(peaks)):
        try:
            # Attempt to calculate HRV indices for the current segment
            HRV_indices = HRV_parameters(peaks=peaks[i], seg_start_idx=clean_segments[i][0], sampling_rate=sampling_rate, window_length=window_length)
        except:
            # Handle exceptions and continue to the next iteration
            if i == 0:
                # If it's the first iteration and an exception occurred, initialize an empty DataFrame
                hrv_data = pd.DataFrame(columns=['Start_idx', 'HR', 'HRV_MeanNN','HRV_SDNN','HRV_RMSSD','HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN','HRV_TINN','HRV_HTI','ULF','VLF','LF','HF','VHF','LFHF','LFn','HFn','LnHF','HRV_SD1','HRV_SD2','HRV_SD1SD2','HRV_S'])                    
            continue
        
        if i == 0:
            # If it's the first iteration, assign the HRV indices to the DataFrame
            hrv_data = HRV_indices
        else:
            # Append HRV indices to the DataFrame
            hrv_data.loc[len(hrv_data.index)] = HRV_indices.values.tolist()[0]
            
    return hrv_data




if __name__ == "__main__":
    # Import a sample data
    FILE_NAME = "201902020222_Data.csv"
    SAMPLING_FREQUENCY = 20
    input_sig = get_data(file_name=FILE_NAME)
    
    # Bandpass filter parameters
    lowcut = 0.5  # Lower cutoff frequency in Hz
    highcut = 3  # Upper cutoff frequency in Hz
    
    # Apply bandpass filter
    filtered_sig = bandpass_filter(sig=input_sig, fs=SAMPLING_FREQUENCY, lowcut=lowcut, highcut=highcut)

    # Run PPG signal quality assessment.
    clean_indices, noisy_indices = ppg_sqa(sig=filtered_sig, sampling_rate=SAMPLING_FREQUENCY)
    
    execfile('GAN.py')
    reconstruction_model_parameters = [G, device]
    
    # Run PPG reconstruction
    ppg_signal, clean_indices, noisy_indices = ppg_reconstruction(sig=filtered_sig, clean_indices=clean_indices, noisy_indices=noisy_indices, sampling_rate=SAMPLING_FREQUENCY, generator=G, device=device)
    
    # Define a window length for clean segments extraction (in seconds)
    WINDOW_LENGTH_SEC = 90
    
    # Calculate the window length in terms of samples
    window_length = WINDOW_LENGTH_SEC*SAMPLING_FREQUENCY
    
    # Scan clean parts of the signal and extract clean segments with the specified window length
    clean_segments = clean_segments_extraction(sig=ppg_signal, noisy_indices=noisy_indices, window_length=window_length)
    
    # Display results
    print("Analysis Results:")
    print("------------------")
    # Check if clean segments are found, if not, print a message
    if len(clean_segments) == 0:
        print('No clean ' + str(WINDOW_LENGTH_SEC) + ' seconds segment was detected in the signal!')
    else:
        # Print the number of clean segments found
        print(str(len(clean_segments)) + ' clean ' + str(WINDOW_LENGTH_SEC) + ' seconds segments was detected in the signal!' )
        
        # Run PPG Peak detection
        peaks, sampling_rate_new = peak_detection(clean_segments, SAMPLING_FREQUENCY)
        
        # Update window length based on the new sampling rate
        window_length_new = WINDOW_LENGTH_SEC*sampling_rate_new
        
        # Perform HR and HRV extraction
        hrv_data = hrv_extraction(clean_segments=clean_segments, peaks=peaks, sampling_rate=sampling_rate_new, window_length=window_length_new)
        print("HR and HRV parameters:")
        print(hrv_data)


