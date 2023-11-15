# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:02:33 2022

@author: mofeli
"""

# Import necessary library
import pandas as pd

# Define a function to extract clean PPG segments based on quality information
def clean_ppg_extraction(ppg_signal, gaps, window_length_min, sample_rate, timestamp):
    """
    Extract clean PPG segments from the input signal based on quality information.
    
    Parameters:
    - ppg_signal (numpy.ndarray): PPG signal.
    - gaps (list): List of noisy segments in the signal.
    - window_length_min (float): Length of the clean PPG segments in minutes.
    - sample_rate (int): Sampling rate of the PPG signal.
    - timestamp (numpy.ndarray): Timestamps corresponding to the PPG signal.
    
    Returns:
    - clean_segments (list): List of clean PPG segments.
    - start_timestamp_segments (list): List of corresponding start timestamps for clean segments.
    """
    
    # Define a nested function to find clean parts in the quality list
    def find_clean_parts(quality_lst):
        # returns clean indexes (start and end) in a list of tuples
        # in the quality list, 0 indicates clean and 1 indicates noisy
        result = []
        start = 0
        for i in range(len(quality_lst)-1):
            if quality_lst[start] == quality_lst[i+1]:
                if i+1 == len(quality_lst)-1:
                    end = i+1
                    if quality_lst[start] == 0:
                        result.append((start,end))
                else:
                    continue
                
            else:
                end = i
                if quality_lst[start] == 0:
                    result.append((start,end))
                    
                start = i+1
        
        return result
    
    # Create a new DataFrame to store timestamp, PPG, and quality information
    new_data = pd.DataFrame(columns=['timestamp','ppg','quality'])
    new_data['timestamp'] = timestamp
    flat_list_gap = [item for gap in gaps for item in gap]
    quality = [1 if i in flat_list_gap else 0 for i in range(len(ppg_signal))]
    new_data['quality'] = quality
    new_data['ppg'] = ppg_signal
    
    # Find clean indexes in the quality list
    clean_indexes = find_clean_parts(new_data['quality'].tolist())
    
    # Calculate the window length in terms of samples
    window_length = window_length_min*60*sample_rate
    
    # Initialize lists to store clean segments and start timestamps
    clean_segments = []
    start_timestamp_segments = []
    
    # Extract clean segments based on window length
    for item in clean_indexes:
        if (item[1]-item[0]) >= window_length:
            ppg_clean = new_data.iloc[item[0]:item[1]][['timestamp', 'ppg']]
            index = item[0]
            # Iterate through the clean segment with the specified window length
            while(index<item[1]):
                clean_segment = ppg_clean.loc[index:index+window_length-1]['ppg']
                start_timestamp = ppg_clean.loc[index]['timestamp']
                
                # Check if the clean segment has the required length
                if len(clean_segment) == window_length:
                    clean_segments.append(clean_segment)
                    start_timestamp_segments.append(start_timestamp)
                index = index + window_length
                
                
    return clean_segments, start_timestamp_segments