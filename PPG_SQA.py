# -*- coding: utf-8 -*-

import pickle
import os
from typing import Tuple, List
from scipy import stats, signal
import more_itertools as mit
import joblib
from utils import normalize_data, get_data, bandpass_filter, find_peaks, resample_signal
import warnings
import numpy as np


warnings.filterwarnings("ignore")

MODEL_PATH = "models"
SCALER_FILE_NAME = "Train_data_scaler.save"
SQA_MODEL_FILE_NAME = 'OneClassSVM_model.sav'
MODEL_SAMPLING_FREQUENCY = 20
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
    
    Args:
        sig: Input signal (e.g., PPG).
        sig_indices: Corresponding indices for the input signal.
        sampling_rate: Sampling rate of the PPG signal.
        method: Segmentation method. Options: 'standard' or 'shifting'.
            Segments do not overlap for 'standard' and overlap with the
            size of (segment_size - shift_size) for 'shifting'.
        segment_size: Size of the segment (in second).
        shift_size: Size of the shift (in seconds) in segmentation
            in case method is 'shifting'.
    
    Return:
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
    
    Args:
        ppg: Input PPG signal.
        sampling_rate: Sampling rate of the PPG signal.
    
    Return:
        hc: List of heart cycles
    """
    # Normalization
    ppg_normalized = normalize_data(ppg)

    # Upsampling signal by 2
    sampling_rate = sampling_rate*2
    ppg_upsampled = signal.resample(ppg_normalized, len(ppg_normalized)*2)

    # Systolic peak detection
    peaks, ppg_cleaned = find_peaks(
        ppg=ppg_upsampled, sampling_rate=sampling_rate, return_sig=True)

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
    
    Args:
        hc: List of heart cycles
    
    Return:
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
    
    Args:
        hc: List of heart cycles
    
    Return:
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
    
    Args:
        ppg: Input PPG signal.
        sampling_rate: Sampling rate of the PPG signal.
    
    Return:
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


def sqa(
        sig: np.ndarray,
        sampling_rate: int,
        filter_signal: bool = True,
) -> Tuple[list, list]:
    """
    Perform PPG Signal Quality Assessment (SQA).
    
    This function assesses the quality of a PPG signal by classifying its segments
    as reliable (clean) or unrelaible (noisy) using a pre-trained model.

    The clean indices represent parts of the PPG signal that are deemed reliable,
    while the noisy indices indicate parts that may be affected by noise or artifacts.
    
    Args:
        sig (np.ndarray): PPG signal.
        sampling_rate (int): Sampling rate of the PPG signal.
        filter_signal (bool): True if the signal has not filtered using
            a bandpass filter.
    
    Return:
        clean_indices: A list of clean indices.
        noisy_indices: A list of noisy indices.
        
    
    Reference:
        Feli, M., Azimi, I., Anzanpour, A., Rahmani, A. M., & Liljeberg, P. (2023).
        An energy-efficient semi-supervised approach for on-device photoplethysmogram signal quality assessment. 
        Smart Health, 28, 100390.


    """
    # Load pre-trained model and normalization scaler
    scaler = joblib.load(os.path.join(MODEL_PATH, SCALER_FILE_NAME))
    model = pickle.load(
        open(os.path.join(MODEL_PATH, SQA_MODEL_FILE_NAME), 'rb'))
    
    resampling_flag = False
    # Check if resampling is needed and perform resampling if necessary
    if sampling_rate != MODEL_SAMPLING_FREQUENCY:
        sig = resample_signal(
            sig=sig, fs_origin=sampling_rate, fs_target=MODEL_SAMPLING_FREQUENCY)
        resampling_flag = True
        resampling_rate = sampling_rate/MODEL_SAMPLING_FREQUENCY
        sampling_rate = MODEL_SAMPLING_FREQUENCY

    # Apply bandpass filter if needed
    if filter_signal:
        sig = bandpass_filter(
            sig=sig, fs=sampling_rate, lowcut=0.5, highcut=3)

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
    noisy_indices = [noisy_indices[i] for i in range(
        len(noisy_indices)) if len(noisy_indices[i]) > SHIFTING_SIZE]
    
    # If resampling performed, update indices according to the original sampling rate
    if resampling_flag:
        clean_indices = [int(index * resampling_rate) for index in clean_indices]
        noisy_indices = [[int(index * resampling_rate) for index in noise] for noise in noisy_indices]


    return clean_indices, noisy_indices


if __name__ == "__main__":
    # Import a sample data
    file_name = "201902020222_Data.csv"
    input_sig = get_data(file_name=file_name)
    input_sampling_rate = 20

    # Run PPG signal quality assessment.
    clean_ind, noisy_ind = sqa(sig=input_sig, sampling_rate=input_sampling_rate)

    # Display results
    print("Analysis Results:")
    print("------------------")
    print(f"Length of the clean signal (in seconds): {len(clean_ind)/input_sampling_rate:.2f}")
    print(f"Number of noisy parts in the signal: {len(noisy_ind)}")

    if len(noisy_ind) > 0:
        print("Length of each noise in the signal (in seconds):")
        for noise in noisy_ind:
            print(f"   - {len(noise)/input_sampling_rate:.2f}")
    else:
        print("The input signal is completely clean!")

