# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:07:50 2022

@author: mofeli
"""
# Import necessary libraries and modules
import pickle
import os
from typing import Tuple
import numpy as np
from scipy import stats
from scipy import signal
import neurokit2 as nk
import more_itertools as mit
import joblib
from utils import normalize_data


MODEL_PATH = "models"
SCALER_FILE_NAME = "Train_data_scaler.save"
SQA_MODEL_FILE_NAME = 'OneClassSVM_model.sav'


def segmentation(
    ppg: np.ndarray,
    ppg_x: np.ndarray,
    sampling_rate: int,
    method: str = 'shifting',
    segmentation_step: int = 5
) -> Tuple[list, list]:
    """
    Segments the PPG signal into fixed-size segments.
    
    Input parameters:
        ppg: PPG signal.
        ppg_x: Corresponding indices for the PPG signal.
        sampling_rate: Sampling rate of the PPG signal.
        method: Segmentation method. Options: 'standard' or 'shifting'.
        segmentation_step: Size of the segmentation step in seconds.
    
    Returns:
        segments: List of PPG signal segments.
        segments_x: List of corresponding indices (time) for the segments.
    """
    segment_size = 30*sampling_rate
    segments = []
    segments_x = []
    index = 0
    while index<len(ppg):
        segment = ppg[index:index+segment_size]
        segment_x = ppg_x[index:index+segment_size]
        if len(segment) == segment_size:
            segments.append(segment)
            segments_x.append(segment_x)
        if method == 'shifting':
            index = index + segmentation_step
        elif method == 'standard':
            index = index + segment_size
    return (segments, segments_x)


# Function to detect heart cycles in the PPG signal
def heart_cycle_detection(sample_rate, upsampling_rate, sample):
    
    sampling_rate = sample_rate * upsampling_rate
    
    
    
    # normalization
    ppg_normed = normalize_data(sample)
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

# Function to extract features from PPG segments
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


# Main function for PPG Signal Quality Assessment
def ppg_sqa(
        sig: np.ndarray,
        sampling_rate: int,
) -> Tuple[list, list]:
    """
    PPG Signal Quality Assessment.
    
    Input parameters:
        sig: PPG signal.
        sampling_rate: Sampling rate of the PPG signal.
    
    Returns:
        x_reliable: ...
        gaps: ...
    """
    # Load pre-trained model and normalization scaler
    scaler = joblib.load(os.path.join(MODEL_PATH, SCALER_FILE_NAME))
    model = pickle.load(
        open(os.path.join(MODEL_PATH, SQA_MODEL_FILE_NAME), 'rb'))

    # Set the segmentation step for 5 seconds
    segmentation_step = 5*sampling_rate

    # Generate indices for the PPG signal
    sig_indices = np.arange(sig.size)

    # Segmentation of the PPG signal
    segments, segments_x = segmentation(
        ppg=sig,
        ppg_x=sig_indices,
        sampling_rate=sampling_rate,
        method='shifting',
        segmentation_step=segmentation_step
    )

    # Initialize lists to store reliable and unreliable segments
    reliable_segments = []
    unreliable_segments = []
    reliable_segments_x = []
    unreliable_segments_x = []

    # Loop through the segments for feature extraction and classification
    for idx in range(len(segments)):
        # Feature extraction
        features = feature_extraction(segments[idx], sampling_rate)

        # Classification
        if(np.isnan(np.array(features)).any()):
            pred = 1
        else:
            features_norm  = scaler.transform([features])
            pred = model.predict(features_norm)

        # Categorize segments based on classification result
        if pred == 0:
            reliable_segments.append(segments[i])
            reliable_segments_x.append(segments_x[i])
        else:
            unreliable_segments.append(segments[i])
            unreliable_segments_x.append(segments_x[i])

    # Generate lists of reliable and unreliable x values        
    x_reliable = list(set([item for segment in reliable_segments_x for item in segment]))
    x_unreliable = [item for item in sig_indices if item not in x_reliable]

    # Extract gaps (noisy parts) of the signal
    gaps = []
    for group in mit.consecutive_groups(x_unreliable):
        gaps.append(list(group))
    gaps = [gaps[i] for i in range(len(gaps)) if len(gaps[i]) > segmentation_step]
    return (x_reliable, gaps)
