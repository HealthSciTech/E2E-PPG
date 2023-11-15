# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:09:18 2022

@author: mofeli
"""

# Import necessary libraries
import neurokit2 as nk
import pandas as pd

# Define a function to extract HRV parameters from PPG segments
def HRV_parameters(ppg_segment, timestamp, peaks, sample_rate, ppg_length):
    """
    Extract HRV parameters from a PPG segment.

    Parameters:
    - ppg_segment (numpy.ndarray): PPG segment.
    - timestamp (float): Timestamp corresponding to the PPG segment.
    - sample_rate (int): Sampling rate of the PPG signal.
    - window_length (int): Length of the PPG segment in samples.

    Returns:
    - HRV_indices (pandas.DataFrame): DataFrame containing HRV indices.
    """
        
    # Calculate pulse rate
    pulse_rate = (len(peaks)/ppg_length)
    
    HRV_indices = []
    
    # HRV values: time, frequency, and non-linear features
    hrv_times = nk.hrv_time(peaks, sampling_rate=sample_rate, show=False)
    hrv_times = hrv_times[['HRV_MeanNN','HRV_SDNN','HRV_RMSSD','HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN','HRV_TINN','HRV_HTI']]
    HRV_indices.append(hrv_times)

    hrv_freqs = nk.hrv_frequency(peaks, sampling_rate=sample_rate, show=False, psd_method="welch")
    HRV_indices.append(hrv_freqs)
    
    hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=sample_rate, show=False)
    hrv_nonlinear = hrv_nonlinear[['HRV_SD1','HRV_SD2','HRV_SD1SD2','HRV_S']]
    HRV_indices.append(hrv_nonlinear)

    # Concatenate HRV indices into a DataFrame
    HRV_indices = pd.concat(HRV_indices, axis=1)
    
    # Add pulse rate and timestamp to the DataFrame
    HRV_indices['HR'] = pulse_rate
    HRV_indices['Timestamp'] = timestamp
    
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



# Define a function to extract HRV parameters from a list of PPG segments
def PPG_HRV_Extraction(ppg_cleans, timestamps, peaks, sample_rate, window_length):
    """
    Extract HRV parameters from a list of PPG segments.

    Parameters:
    - ppg_cleans (list): List of cleaned PPG segments.
    - timestamps (numpy.ndarray): Timestamps corresponding to the PPG segments.
    - sample_rate (int): Sampling rate of the PPG signal.
    - window_length (int): Length of the PPG segments in samples.

    Returns:
    - hrv_data (pandas.DataFrame): DataFrame containing HRV indices for all segments.
    """
    
    # Iterate through PPG segments and extract HRV parameters
    for i in range(len(ppg_cleans)):
        try:
            HRV_indices = HRV_parameters(ppg_cleans[i], timestamps[i], peaks[i], sample_rate, window_length)
        except:
            # If an error occurs during HRV extraction, skip to the next segment
            if i == 0:
                hrv_data = pd.DataFrame(columns=['HRV_MeanNN','HRV_SDNN','HRV_RMSSD','HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_Prc80NN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_MinNN', 'HRV_MaxNN','HRV_TINN','HRV_HTI','ULF','VLF','LF','HF','VHF','LFHF','LFn','HFn','LnHF','HRV_SD1','HRV_SD2','HRV_SD1SD2','HRV_S','HR','Timestamp'])                    
            continue
        if i == 0:
            hrv_data = HRV_indices
        else:
            # Append HRV indices to the DataFrame
            hrv_data.loc[len(hrv_data.index)] = HRV_indices.values.tolist()[0]
        # hrv_parameters = [pulse_rate]
        # hrv_parameters = hrv_parameters + hrv_times.values.tolist()[0]
        # hrv_parameters = hrv_parameters + hrv_freqs.values.tolist()[0]
        # hrv_parameters = hrv_parameters + hrv_nonlinear.values.tolist()[0]
        # HRV_indices.loc[len(HRV_indices.index)] = hrv_parameters
        
    return hrv_data


