# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import numpy as np
from utils import get_data
from ppg_sqa import sqa
from ppg_reconstruction import reconstruction
warnings.filterwarnings("ignore")

def clean_seg_extraction(
    sig: np.ndarray,
    noisy_indices: list,
    window_length: int
) -> list:
    
    """
    Scan the clean parts of the signal and extract clean segments based on the input window length.
    
    Args:
        sig (numpy.ndarray): Input PPG signal.
        noisy_indices (list): List of noisy segment indices.
        window_length (int): Desired window length for clean segment extraction in terms of samples.
        
    Return:
        clean_segments (list): List of clean PPG segments with the specified window length and their starting index.
    """
    
    def find_clean_parts(quality_lst:list) -> list:
        '''
        Scan the quality vector and find the start and end indices of clean parts.

        Args:
            quality_lst (list): Quality vector of the signal (0 indictes clean and 1 indicates noisy)
                
        Return:
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
    file_name = "201902020222_Data.csv"
    input_sampling_rate = 20
    input_sig = get_data(file_name=file_name)

    # Define a window length for clean segments extraction (in seconds)
    window_length_sec = 90

    # Run PPG signal quality assessment.
    clean_ind, noisy_ind = sqa(sig=input_sig, sampling_rate=input_sampling_rate)


    # Run PPG reconstruction
    reconstructed_signal, clean_ind, noisy_ind = reconstruction(
        sig=input_sig,
        clean_indices=clean_ind,
        noisy_indices=noisy_ind,
        sampling_rate=input_sampling_rate)


    # Calculate the window length in terms of samples
    window_length = window_length_sec*input_sampling_rate
    
    # Scan clean parts of the signal and extract clean segments with the specified window length
    clean_segments = clean_seg_extraction(sig=reconstructed_signal, noisy_indices=noisy_ind, window_length=window_length)
    
    # Display results
    print("Analysis Results:")
    print("------------------")
    # Check if clean segments are found, if not, print a message
    if len(clean_segments) == 0:
        print('No clean ' + str(window_length_sec) + ' seconds segment was detected in the signal!')
    else:
        # Print the number of detected clean segments
        print(str(len(clean_segments)) + ' clean ' + str(window_length_sec) + ' seconds segments was detected in the signal!' )
        print("Starting index of each segment in seconds:")
        for seg in clean_segments:
            print(int(seg[0] / input_sampling_rate))
    