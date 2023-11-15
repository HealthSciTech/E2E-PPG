# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:02:25 2023

@author: mofeli
"""
from scipy.signal import butter, filtfilt

def PPG_bandpass_filter(ppg, lowcut, highcut, sample_rate):
    
    
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y
    
    
    ppg_filtered = butter_bandpass_filter(ppg, lowcut, highcut, sample_rate)
    
    return ppg_filtered

    
    