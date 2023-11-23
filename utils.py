
# -*- coding: utf-8 -*-
'''
Miscellaneous functions

'''

from typing import List, Tuple
import os
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
import neurokit2 as nk


def get_data(
        file_name: str,
        local_directory: str = "data",
        usecols: List[str] = ['ppg'],
) -> np.ndarray:
    """
    Import data (e.g., PPG signals)
    
    Args:
        file_name (str): Name of the input file
        local_directory (str): Data directory
        usecols (List[str]): The columns to read from the input file
    
    Return:
        sig (np.ndarray): the input signal (e.g., PPG)
    """
    try:
        # Construct the file path
        file_path = os.path.join(local_directory, file_name)
        # Load data from the specified CSV file
        input_data = pd.read_csv(
            file_path,
            delim_whitespace=True,
            usecols=usecols)
        # Extract signal
        sig = input_data[usecols[0]].values
        return sig
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except pd.errors.EmptyDataError:
        print(f"Empty data in file: {file_name}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # Return None in case of an error
    return None


def normalize_data(sig: np.ndarray) -> np.ndarray:
    """
    Normalize the input signal between zero and one
    
    Args:
        sig (np.ndarray): PPG signal.
    
    Return:
        np.ndarray: Normalized signal
    """
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))


def resample_signal(
        sig: np.ndarray,
        fs_origin: int,
        fs_target: int = 20,
) -> np.ndarray:
    """
    Resample the signal

    Args:
        sig (np.ndarray): The input signal.
        fs_origin (int): The sampling frequency of the input signal.
        fs_target (int): The sampling frequency of the output signal.

    Return:
        sig_resampled (np.ndarray): The resampled signal.
    """
    # Exit if the sampling frequency already is 20 Hz (return the original signal)
    if fs_origin == fs_target:
        return sig
    # Calculate the resampling rate
    resampling_rate = 20/fs_origin
    # Resample the signal
    sig_resampled = resample(sig, int(len(sig)*resampling_rate))
    # Update the sampling frequency
    return sig_resampled


def bandpass_filter(
        sig: np.ndarray,
        fs: int,
        lowcut: float,
        highcut: float,
        order: int=2
) -> np.ndarray:
    """
    Apply a bandpass filter to the input signal.

    Args:
        sig (np.ndarray): The input signal.
        fs (int): The sampling frequency of the input signal.
        lowcut (float): The low cutoff frequency of the bandpass filter.
        highcut (float): The high cutoff frequency of the bandpass filter.

    Return:
        sig_filtered (np.ndarray): The filtered signal using a Butterworth bandpass filter.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    sig_filtered = filtfilt(b, a, sig)
    return sig_filtered


def find_peaks(
        ppg: np.ndarray,
        sampling_rate: int,
        return_sig: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in PPG.

    Args:
        ppg (np.ndarray): The input PPG signal.
        sampling_rate (int): The sampling rate of the signal.
        return_sig (bool): If True, return the cleaned PPG
            signal along with the peak indices (default is False).

    Return:
        peaks (np.ndarray): An array containing the indices of
            the detected peaks in the PPG signal.
        ppg_cleaned (np.ndarray): The cleaned PPG signal, return if return_sig is True.

    """

    # Clean the PPG signal and prepare it for peak detection
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)

    # Peak detection
    info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)
    peaks = info["PPG_Peaks"]

    # Return either just the peaks or both the cleaned signal and peaks
    if return_sig:
        return peaks, ppg_cleaned
    else:
        return peaks, None
