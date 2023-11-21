# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from utils import get_data, bandpass_filter, check_and_resample
from ppg_sqa import sqa
from ppg_reconstruction import reconstruction
import warnings
warnings.filterwarnings("ignore")

def clean_seg_extraction(
        sig: np.ndarray,
        noisy_indices: list,
        window_length: int) -> list:
    
    """
    Scan the clean parts of the signal and extract clean segments based on the input window length.
    
    Input parameters:
        sig (numpy.ndarray): Input PPG signal.
        noisy_indices (list): List of noisy segment indices.
        window_length (int): Desired window length for clean segment extraction in terms of samples.
        
    Returns:
        clean_segments (list): List of clean PPG segments with the specified window length and their starting index.
    """
    
    def find_clean_parts(quality_lst:list) -> list:
        '''
        Scan the quality vector and find the start and end indices of clean parts.

        Input parameters:
            quality_lst (list): Quality vector of the signal (0 indictes clean and 1 indicates noisy)
                
        Returns
            start_end_clean (list): Start and end indices of the clean parts in a list of tuples
        '''
        
        start_end_clean = []
        start = 0
        for i in range(len(quality_lst)-1):
            if quality_lst[start] == quality_lst[i+1]:
                if i+1 == len(quality_lst)-1:
                    end = i+1
                    if quality_lst[start] == 0:
                        start_end_clean.append((start,end))
                else:
                    continue
                
            else:
                end = i
                if quality_lst[start] == 0:
                    start_end_clean.append((start,end))
                    
                start = i+1
        
        return start_end_clean
    
    
    # Create a new DataFrame to store PPG, and quality information
    quality_df = pd.DataFrame(columns=['ppg','quality'])
    
    # Flatten the noise indices list
    flat_list_noise = [item for noise in noisy_indices for item in noise]
    
    # Define a quality vector (0 indictes clean and 1 indicates noisy)
    quality = [1 if i in flat_list_noise else 0 for i in range(len(sig))]
    
    # Store ppg signal with quality vector in dataframe
    quality_df['quality'] = quality
    quality_df['ppg'] = sig
    
    # Find start and end indices of clean parts in the quality list
    start_end_clean_idx = find_clean_parts(quality_df['quality'].tolist())
    
    # Initialize a list to store total clean segments with the specified window length
    clean_segments = []

    # Extract clean segments based on window length
    for indices in start_end_clean_idx:
        # Check if the current clean part has the required window length
        if (indices[1] - indices[0]) >= window_length:
            # Select the current clean part
            clean_part = quality_df['ppg'][indices[0] : indices[1]].tolist()
            
            # Calculate the number of segments with the specified window length that can be extarcted from the current clean part
            num_segments = len(clean_part) // window_length
            
            # Extract clean segment with the specified window length from current clean part and their starting indices
            segments = [((indices[0] + i * window_length), clean_part[i * window_length: (i + 1) * window_length]) for i in range(num_segments)]
            
            # Add extracted segments to total clean segments
            clean_segments.extend(segments)

                
    return clean_segments




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
        print("Starting index of each segment in seconds:")
        for seg in clean_segments:
            print(int(seg[0] / sampling_rate))
    
    
    
    
    
    