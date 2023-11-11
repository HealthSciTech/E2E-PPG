# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:02:33 2022

@author: mofeli
"""
import pandas as pd

def clean_ppg_extraction(ppg_signal, gaps, window_length_min, sample_rate, timestamp):
     
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
    
    
    new_data = pd.DataFrame(columns=['timestamp','ppg','quality'])
    new_data['timestamp'] = timestamp
    flat_list_gap = [item for gap in gaps for item in gap]
    quality = [1 if i in flat_list_gap else 0 for i in range(len(ppg_signal))]
    new_data['quality'] = quality
    new_data['ppg'] = ppg_signal
    

    clean_indexes = find_clean_parts(new_data['quality'].tolist())
    
    
    window_length = window_length_min*60*sample_rate
    
    clean_segments = []
    start_timestamp_segments = []
    for item in clean_indexes:
        if (item[1]-item[0]) >= window_length:
            ppg_clean = new_data.iloc[item[0]:item[1]][['timestamp', 'ppg']]
            index = item[0]
            while(index<item[1]):
                clean_segment = ppg_clean.loc[index:index+window_length-1]['ppg']
                start_timestamp = ppg_clean.loc[index]['timestamp']
                if len(clean_segment) == window_length:
                    clean_segments.append(clean_segment)
                    start_timestamp_segments.append(start_timestamp)
                index = index + window_length
                
                
    return clean_segments, start_timestamp_segments