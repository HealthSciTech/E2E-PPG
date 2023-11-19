# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:07:50 2022

@author: mofeli
"""
# Import necessary libraries and modules
import pickle
import os
from typing import Tuple, List
import numpy as np
from scipy import stats
from scipy import signal
import more_itertools as mit
import joblib
from utils import normalize_data, get_data, bandpass_filter, find_peaks

MODEL_PATH = "models"
SCALER_FILE_NAME = "Train_data_scaler.save"
SQA_MODEL_FILE_NAME = 'OneClassSVM_model.sav'
SEGMENT_SIZE = 30
SHIFTING_SIZE = 2


def segmentation(
    sig: np.ndarray,
    sig_indices: np.ndarray,
    sampling_rate: int,
    method: str = 'shifting',
    segment_size: int = 30,
    shift_size: int = 2,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Segments the signals (PPG) and their indices into fixed-size segments.
    
    Input parameters:
        sig: Input signal (e.g., PPG).
        sig_indices: Corresponding indices for the input signal.
        sampling_rate: Sampling rate of the PPG signal.
        method: Segmentation method. Options: 'standard' or 'shifting'.
            Segments do not overlap for 'standard' and overlap with the
            size of (segment_size - shift_size) for 'shifting'.
        segment_size: Size of the segment (in second).
        shift_size: Size of the shift (in seconds) in segmentation
            in case method is 'shifting'.
    
    Returns:
        segments_sig: List of segments (PPG).
        segments_indices: List of segments (indices).
    """
    signal_length = len(sig)
    segment_length = int(segment_size*sampling_rate)
    shift_length = int(shift_size*sampling_rate)
    if method == 'standard':
        # Non-overlapping segments
        segments_sig = [sig[i:i+segment_length] for i in range(
            0, signal_length, segment_length
                ) if i + segment_length <= signal_length]
        segments_indices = [sig_indices[i:i+segment_length] for i in range(
            0, signal_length, segment_length
                ) if i + segment_length <= signal_length]
    elif method == 'shifting':
        # Overlapping segments
        segments_sig = [sig[i:i+segment_length] for i in range(
            0, signal_length - segment_length + 1, shift_length
                ) if i + segment_length <= signal_length]
        segments_indices = [sig_indices[i:i+segment_length] for i in range(
            0, signal_length - segment_length + 1, shift_length
                ) if i + segment_length <= signal_length]
    else:
        raise ValueError("Invalid method. Use 'standard' or 'shifting'.")
    return segments_sig, segments_indices


def heart_cycle_detection(
        ppg: np.ndarray,
        sampling_rate: int,
) -> list:
    """
    Extract heart cycles from the PPG signal
    
    Input parameters:
        ppg: Input PPG signal.
        sampling_rate: Sampling rate of the PPG signal.
    
    Returns:
        hc: List of heart cycles
    """
    # Normalization
    ppg_normalized = normalize_data(ppg)
    # Upsampling signal by 2
    sampling_rate = sampling_rate*2
    ppg_upsampled = signal.resample(ppg_normalized, len(ppg_normalized)*2)
    
    # Systolic peak detection
    ppg_cleaned, peaks = find_peaks(ppg=ppg_upsampled, sampling_rate=sampling_rate, return_sig=True)

    # Heart cycle detection based on the peaks and fixed intervals
    hc = []
    if len(peaks) < 2:
        return hc
    # Define a fixed interval in PPG signal to detect heart cycles
    beat_bound = round((len(ppg_upsampled)/len(peaks))/2)
    # Ignore the first and last beat to prevent boundary error
    for i in range(1, len(peaks) - 1):
        # Select beat from the signal and add it to the list
        beat_start = peaks[i] - beat_bound
        beat_end = peaks[i] + beat_bound
        if beat_start >= 0 and beat_end < len(ppg_cleaned):
            beat = ppg_cleaned[beat_start:beat_end]
            if len(beat) >= beat_bound*2:
                hc.append(beat)
    return hc


def energy_hc(hc: list) -> float:
    """
    Extract energy of heart cycle
    
    Input parameters:
        hc: List of heart cycles
    
    Returns:
        var_energy: Variation of heart cycles energy
    """
    energy = []
    for beat in hc:
        energy.append(np.sum(beat*beat))
    if not energy:
        var_energy = 0
    else:
        # Calculate variation
        var_energy = max(energy) - min(energy)
    return var_energy


def template_matching_features(hc: list) -> Tuple[float, float]:
    """
    Extract template matching features from heart cycles
    
    Input parameters:
        hc: List of heart cycles
    
    Returns:
        tm_ave_eu: Average of Euclidean distance with the template
        tm_ave_corr: Average of correlation with the template
    """
    hc = np.array([np.array(xi) for xi in hc if len(xi) != 0])
    # Calculate the template by averaging all heart cycles
    template = np.mean(hc, axis=0)
    # Euclidean distance and correlation
    distances = []
    corrs = []
    for beat in hc:
        distances.append(np.linalg.norm(template-beat))
        corr_matrix = np.corrcoef(template, beat)
        corrs.append(corr_matrix[0, 1])
    tm_ave_eu = np.mean(distances)
    tm_ave_corr = np.mean(corrs)
    return tm_ave_eu, tm_ave_corr


def feature_extraction(
        ppg: np.ndarray,
        sampling_rate: int,
) -> List[float]:
    """
    Extract features from PPG signal
    
    Input parameters:
        ppg: Input PPG signal.
        sampling_rate: Sampling rate of the PPG signal.
    
    Returns:
        features: List of features
    """
    # feature 1: Interquartile range
    iqr_rate = stats.iqr(ppg, interpolation='midpoint')

    # feature 2: STD of power spectral density
    _, pxx_den = signal.periodogram(ppg, sampling_rate)
    std_p_spec = np.std(pxx_den)

    # Heart cycle detection
    hc = heart_cycle_detection(ppg=ppg, sampling_rate=sampling_rate)
    if hc:
        # feature 3: variation in energy of heart cycles
        var_energy = energy_hc(hc)

        # features 4, 5: average Euclidean and Correlation in template matching
        tm_ave_eu, tm_ave_corr = template_matching_features(hc)
    else:
        var_energy = np.nan
        tm_ave_eu = np.nan
        tm_ave_corr = np.nan
    features = [iqr_rate, std_p_spec, var_energy, tm_ave_eu, tm_ave_corr]
    return features


# Main function for PPG Signal Quality Assessment
def ppg_sqa(
        sig: np.ndarray,
        sampling_rate: int,
) -> Tuple[list, list]:
    """
    Perform PPG Signal Quality Assessment (SQA).
    
    Input parameters:
        sig (np.ndarray): PPG signal.
        sampling_rate (int): Sampling rate of the PPG signal.
    
    Returns:
        clean_indices: A list of clean indices.
        noisy_indices: A list of noisy indices.
        
        
    This function assesses the quality of a PPG signal by classifying its segments
    as reliable (clean) or unrelaible (noisy) using a pre-trained model.

    The clean indices represent parts of the PPG signal that are deemed reliable,
    while the noisy indices indicate parts that may be affected by noise or artifacts.


    """
    # Load pre-trained model and normalization scaler
    scaler = joblib.load(os.path.join(MODEL_PATH, SCALER_FILE_NAME))
    model = pickle.load(
        open(os.path.join(MODEL_PATH, SQA_MODEL_FILE_NAME), 'rb'))

    # Generate indices for the PPG signal
    sig_indices = np.arange(len(sig))

    # Segment the PPG signal into
    segments, segments_indices = segmentation(
        sig=sig,
        sig_indices=sig_indices,
        sampling_rate=sampling_rate,
        method='shifting',
        segment_size=SEGMENT_SIZE,
        shift_size=SHIFTING_SIZE,
    )

    # Initialize lists to store all reliable and unreliable segments
    reliable_segments_all = []
    unreliable_segments_all = []
    reliable_indices_all = []
    unreliable_indices_all = []

    # Loop through the segments for feature extraction and classification
    for idx, segment in enumerate(segments):

        # Feature extraction
        features = feature_extraction(segment, sampling_rate)

        # Classification
        if np.isnan(np.array(features)).any():
            pred = 1
        else:
            features_norm  = scaler.transform([features])
            pred = model.predict(features_norm)

        # Categorize segments based on classification result
        if pred == 0:
            reliable_segments_all.append(segment)
            reliable_indices_all.append(segments_indices[idx])
        else:
            unreliable_segments_all.append(segment)
            unreliable_indices_all.append(segments_indices[idx])

    # Generate flatten lists of reliable indices as clean indices
    clean_indices = list(set([item for segment in reliable_indices_all for item in segment]))
    
    # The indices that dont exist in the flat list of clean indices indicate unreliable indices
    unreliable_indices = [item for item in sig_indices if item not in clean_indices]

    # Unflat the unreliable_indices list to separte noisy parts
    noisy_indices = []
    for group in mit.consecutive_groups(unreliable_indices):
        noisy_indices.append(list(group))
    noisy_indices = [noisy_indices[i] for i in range(len(noisy_indices)) if len(noisy_indices[i]) > SHIFTING_SIZE]
    
    return clean_indices, noisy_indices


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


    # Display results
    print("Analysis Results:")
    print("------------------")
    print(f"Length of the clean signal (in seconds): {len(clean_indices)/SAMPLING_FREQUENCY:.2f}")
    print(f"Number of noisy parts in the signal: {len(noisy_indices)}")
    
    if len(noisy_indices) > 0:
        print("Length of each noise in the signal (in seconds):")
        for noise in noisy_indices:
            print(f"   - {len(noise)/SAMPLING_FREQUENCY:.2f}")