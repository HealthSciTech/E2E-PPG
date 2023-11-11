# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:07:50 2022

@author: mofeli
"""

import heartpy as hp
import numpy as np
from scipy import stats
from scipy import signal
import neurokit2 as nk
import more_itertools as mit
import matplotlib.pyplot as plt
import pickle
import joblib

def segmentation(ppg, ppg_x, sample_rate, method='shifting', segmentation_step=5):
    
    segment_size=30*sample_rate
    segments = []
    segments_x = []
    index = 0
    while(index<len(ppg)):
        segment = ppg[index:index+segment_size]
        segment_x = ppg_x[index:index+segment_size]
        if len(segment) == segment_size:
            segments.append(segment)
            segments_x.append(segment_x)
        if method == 'shifting':
            index = index + segmentation_step
        elif method == 'standard':
            index = index + segment_size
    return segments, segments_x


def heart_cycle_detection(sample_rate, upsampling_rate, sample):
    
    sampling_rate = sample_rate * upsampling_rate
    
    def NormalizeData(signal):
        return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    # normalization
    ppg_normed = NormalizeData(sample)
    # upsampling signal
    resampled = signal.resample(ppg_normed, len(ppg_normed) * upsampling_rate)
    # clean PPG signal and prepare it for peak detection
    ppg_cleaned = nk.ppg_clean(resampled, sampling_rate=sampling_rate)
    # peak detection
    info  = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)
    peaks = info["PPG_Peaks"]
    # heart cycle detection based on the peaks and fixed intervals
    beats = []
    if len(peaks) != 0:
        # define a fixed interval in PPG signal to detect heart cycles
        beat_bound = round((len(resampled)/len(peaks))/2)
        for i in peaks:
            # we ignore the first and last beat to prevent boundary error
            if i == peaks[0]:
                continue
            elif i == peaks[-1]:
                break
            else:
                # select beat from the cleaned signal and add it to the list
                beat = ppg_cleaned[(i-beat_bound):(i+beat_bound)]
                if len(beat) < beat_bound*2:
                    continue
                beats.append(beat)
    return beats

def feature_extraction(ppg_segment, sample_rate):
    
    def energy_hc(sample_beats):
        energy = []
        for i in range(len(sample_beats)):
            energy.append(np.sum(sample_beats[i]*sample_beats[i]))
        if energy == []:
            var_energy = 0
        else:
            # calculate variation
            var_energy = max(energy) - min(energy)

        return var_energy
    
    
    def template_matching_features(sample_beats):

        beats = np.array([np.array(xi) for xi in sample_beats if xi.size != 0])
        # Calculate the template by averaging all beats
        template = np.mean(beats, axis=0)
        # Average Euclidian
        distances = []
        for i in range(len(sample_beats)):
            distances.append(np.linalg.norm(template-sample_beats[i]))
        tm_ave_eu = np.mean(distances)

        # Average Correlation
        corrs = []
        for i in range(len(sample_beats)):
            corr_matrix = np.corrcoef(template, sample_beats[i])
            corrs.append(corr_matrix[0,1])
        tm_ave_corr = np.mean(corrs)


        return tm_ave_eu, tm_ave_corr
    
    # feature 1: Interquartile range  
    iqr_rate = stats.iqr(ppg_segment, interpolation='midpoint')
    
    # feature 2: STD of power spectral density
    f, Pxx_den = signal.periodogram(ppg_segment, sample_rate)
    std_p_spec = np.std(Pxx_den)
    
    # Heart cycle detection
    beats = heart_cycle_detection(sample_rate=sample_rate, upsampling_rate=2, sample=ppg_segment)
    
    
    if len(beats) != 0:
        
        # feature 3: variation in energy of heart cycles
        var_energy = energy_hc(beats)
        
        # features 4, 5: average Euclidean and Correlation in template matching
        tm_ave_eu, tm_ave_corr = template_matching_features(beats)
        
        #Classification
        features = [iqr_rate, std_p_spec, var_energy, tm_ave_eu, tm_ave_corr]
    else:
        features = [iqr_rate, std_p_spec, np.nan, np.nan, np.nan]
    
    
    return features


def PPG_SQA(ppg_filtered, sample_rate, doPlot=False):
    
    # model and normalization scaler
    scaler_filename = "Train_data_scaler.save"
    model_filename = 'OneClassSVM_model.sav'
    scaler = joblib.load(scaler_filename)
    model = pickle.load(open(model_filename, 'rb'))
    
    # 5 sec segmentation step
    segmentation_step = 5*sample_rate
    
    ppg_x = list(range(len(ppg_filtered)))
    # Segmentation
    segments, segments_x = segmentation(ppg_filtered, ppg_x, sample_rate, 'shifting', segmentation_step)

    
    reliable_segments = []
    unreliable_segments = []
    reliable_segments_x = []
    unreliable_segments_x = []
    for i in range(len(segments)):
        
        # Feature extraction
        features = feature_extraction(segments[i], sample_rate)
        
        # Classification
        if(np.isnan(np.array(features)).any()):
            pred = 1
        else:  
            features_norm  = scaler.transform([features])
            pred = model.predict(features_norm)
        
        if pred == 0:
            reliable_segments.append(segments[i])
            reliable_segments_x.append(segments_x[i])
        else:
            unreliable_segments.append(segments[i])
            unreliable_segments_x.append(segments_x[i])
            
            
    x_reliable = list(set([item for segment in reliable_segments_x for item in segment]))
    x_unreliable = [item for item in ppg_x if item not in x_reliable]
    
    # Extract gaps (noisy parts) of the signal
    gaps = []
    for group in mit.consecutive_groups(x_unreliable):
        gaps.append(list(group))
    gaps = [gaps[i] for i in range(len(gaps)) if len(gaps[i]) > segmentation_step]
    
    
    # add 2 min confidnece interval to before and after gap 
    # conf_interval = 2*sample_rate
    # for gap in gaps:
    #     if gap == gaps[0]:
    #         if gap[0] < conf_interval:
    #             gap[0:0] = x_reliable[:gap[0]]
    #             if gap[-1] < len(ppg_filtered)-(2*sample_rate):
    #                 gap.extend(x_reliable[gap[-1]:gap[-1]+conf_interval])
    #             else:
    #                 gap.extend(x_reliable[gap[-1]:)
                                          
                                    
                    
    # if doPlot==True:
    #     x = range(len(ppg_filtered))
    #     # plt.plot(x,ppg_filt)
    #     for i in range(len(gaps)):
    #         plt.figure(figsize=[12,8])
    #         plt.plot(x[gaps[i][0]-200:gaps[i][-1]+200],ppg_filtered[gaps[i][0]-200:gaps[i][-1]+200])
    #         plt.axvspan(gaps[i][0], gaps[i][-1], color='red', alpha=0.5)
#     plt.title(filename)
    
    
    return x_reliable, gaps
