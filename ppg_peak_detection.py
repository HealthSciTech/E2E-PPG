# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:01:36 2023

@author: mofeli
"""

import neurokit2 as nk
import heartpy as hp
import numpy as np
from scipy import signal
from peak_detection import PPG_Peak
from heartpy.datautils import rolling_mean

def peak_detection(ppg_cleans, sample_rate, method='nk'):
    
    
    segments_peaks = []
    
    if method == 'nk':
        # Neurokit method
        
        upsampling_rate = 2
        sample_rate_new = sample_rate * upsampling_rate

        # Normalize PPG signal
        def NormalizeData(signal):
            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        
        # remove 10 samples from the beginning and end of the segment
        # ppg_segment = ppg_segment[10:-10]
        for i in range(len(ppg_cleans)):
            # Normalize PPG signal
            ppg_normed = NormalizeData(ppg_cleans[i])
            # Upsample the signal
            resampled = signal.resample(ppg_normed, len(ppg_normed) * upsampling_rate)
            
            ppg_cleaned = nk.ppg_clean(resampled, sampling_rate=sample_rate_new)
            info  = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sample_rate_new)
            peaks = info["PPG_Peaks"]
            segments_peaks.append(peaks)
        return segments_peaks
    
    elif method == 'kazemi':
        for i in range(len(ppg_cleans)):
            peaks= PPG_Peak(np.asarray(ppg_cleans[i]), sample_rate, seconds = 15, overlap = 0, minlen = 15, doplot = False)
            segments_peaks.append(peaks)
        return segments_peaks

    elif method == 'heartpy':
        # HeartPy method
        for i in range(len(ppg_cleans)):
            rol_mean = rolling_mean(ppg_cleans[i], windowsize = 0.75, sample_rate = sample_rate)
            wd = hp.peakdetection.detect_peaks(ppg_cleans[i], rol_mean, ma_perc = 20, sample_rate = sample_rate)
            peaks = wd['peaklist']
            segments_peaks.append(peaks)
        return segments_peaks
        
    else:
        print("Invalid method. Please choose from 'neurokit', 'kazemi', or 'heartpy'")
        return None

