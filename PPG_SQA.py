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
import neurokit2 as nk
import more_itertools as mit
import joblib
from utils import normalize_data, get_data
from PPG_Filtering import PPG_bandpass_filter

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
    shift_size: int = 5,
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
        beats: List of heart cycles
    """
    # Normalization
    ppg_normalized = normalize_data(ppg)
    # Upsampling signal by 2
    sampling_rate = sampling_rate*2
    ppg_upsampled = signal.resample(ppg_normalized, len(ppg_normalized)*2)
    # Clean PPG signal and prepare it for peak detection
    ppg_cleaned = nk.ppg_clean(ppg_upsampled, sampling_rate=sampling_rate)
    # Systolic peak detection
    info  = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)
    peaks = info["PPG_Peaks"]
    # Heart cycle detection based on the peaks and fixed intervals
    beats = []
    if len(peaks) < 2:
        return beats
    # Define a fixed interval in PPG signal to detect heart cycles
    beat_bound = round((len(ppg_upsampled)/len(peaks))/2)
    # We ignore the first and last beat to prevent boundary error
    for i in range(1, len(peaks) - 1):
        # Select beat from the signal and add it to the list
        beat_start = peaks[i] - beat_bound
        beat_end = peaks[i] + beat_bound
        if beat_start >= 0 and beat_end < len(ppg_cleaned):
            beat = ppg_cleaned[beat_start:beat_end]
            if len(beat) >= beat_bound*2:
                beats.append(beat)
    return beats


def energy_hc(beats: list) -> float:
    """
    Extract energy of heart cycle
    
    Input parameters:
        beats: List of heart cycles
    
    Returns:
        var_energy: Variation of heart cycles energy
    """
    energy = []
    for beat in beats:
        energy.append(np.sum(beat*beat))
    if not energy:
        var_energy = 0
    else:
        # Calculate variation
        var_energy = max(energy) - min(energy)
    return var_energy


def template_matching_features(beats: list) -> Tuple[float, float]:
    """
    Extract template matching features from heart cycles
    
    Input parameters:
        beats: List of heart cycles
    
    Returns:
        tm_ave_eu: Average of Euclidean distance with the template
        tm_ave_corr: Average of correlation with the template
    """
    beats = np.array([np.array(xi) for xi in beats if len(xi) != 0])
    # Calculate the template by averaging all beats
    template = np.mean(beats, axis=0)
    # Euclidean distance and correlation
    distances = []
    corrs = []
    for beat in beats:
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
    beats = heart_cycle_detection(ppg=ppg, sampling_rate=sampling_rate)
    if beats:
        # feature 3: variation in energy of heart cycles
        var_energy = energy_hc(beats)

        # features 4, 5: average Euclidean and Correlation in template matching
        tm_ave_eu, tm_ave_corr = template_matching_features(beats)
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

    # Generate indices for the PPG signal
    sig_indices = np.arange(len(sig))

    # Segment the PPG signal into
    segments, segments_x = segmentation(
        sig=sig,
        sig_indices=sig_indices,
        sampling_rate=sampling_rate,
        method='shifting',
        shift_size=SHIFTING_SIZE,
        segment_size=SEGMENT_SIZE,
    )

    # Initialize lists to store reliable and unreliable segments
    reliable_segments = []
    unreliable_segments = []
    reliable_segments_x = []
    unreliable_segments_x = []

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
            reliable_segments.append(segment)
            reliable_segments_x.append(segments_x[idx])
        else:
            unreliable_segments.append(segment)
            unreliable_segments_x.append(segments_x[idx])

    # Generate lists of reliable and unreliable signals indices
    x_reliable = list(set(
        [item for segment in reliable_segments_x for item in segment]))
    x_unreliable = [item for item in sig_indices if item not in x_reliable]

    # Extract gaps (noisy parts) of the signal
    gaps = []
    for group in mit.consecutive_groups(x_unreliable):
        gaps.append(list(group))
    gaps = [gaps[i] for i in range(len(gaps)) if len(gaps[i]) > SHIFTING_SIZE]
    return (x_reliable, gaps)


if __name__ == "__main__":
    # Import a sample data
    FILE_NAME = "201902020222_Data.csv"
    SAMPLING_FREQUENCY = 20
    input_sig = get_data(file_name=FILE_NAME)
    
    
    # Bandpass filter parameters
    lowcut = 0.5  # Lower cutoff frequency in Hz
    highcut = 3  # Upper cutoff frequency in Hz
    
    # Apply bandpass filter
    filtered_sig = PPG_bandpass_filter(input_sig, lowcut, highcut, SAMPLING_FREQUENCY)

    # Run PPG signal quality assessment.
    x_reliable, gaps = ppg_sqa(sig=filtered_sig, sampling_rate=SAMPLING_FREQUENCY)
    print(x_reliable)
    # print(gaps)
