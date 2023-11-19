
# -*- coding: utf-8 -*-
'''
Miscellaneous functions

'''

import numpy as np
import pandas as pd
from typing import List, Tuple
import os
from scipy.signal import resample
from scipy.signal import butter, filtfilt
import neurokit2 as nk

def get_data(
        file_name: str,
        local_directory: str = "data",
        usecols: List[str] = ['ppg'],
) -> np.ndarray:
    """
    Import data
    
    Input parameters:
        file_name: Name of the input file
        local_directory: Data directory
        usecols: The columns to read from the input file
    
    Returns:
        sig: the input signal (e.g., PPG)
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
    
    Parameters:
        sig: PPG signal.
    
    Returns:
        Normalized signal
    """
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))




def check_and_resample(
        sig: np.ndarray,
        fs: int) -> Tuple[np.ndarray, int]:
    """
    Check if the given signal has a sampling frequency (fs) of 20 Hz.
    If not, resample the signal to 20 Hz.

    Parameters:
    - sig: np.ndarray
        The input signal.
    - fs: int
        The sampling frequency of the input signal.

    Returns:
    - sig_resampled: np.ndarray
        The resampled signal.
    - fs_resampled: int
        The updated sampling frequency.
    """

    # Check if the sampling frequency is not 20 Hz
    if fs != 20:
        # Calculate the resampling rate
        resampling_rate = 20/fs
        # Resample the signal
        sig_resampled = resample(sig, int(len(sig)*resampling_rate))
        # Update the sampling frequency
        fs_resampled = 20
        return sig_resampled, fs_resampled

    # If the sampling frequency is already 20 Hz, return the original signal
    else:
        return sig, fs



def bandpass_filter(
        sig: np.ndarray,
        fs: int,
        lowcut: float,
        highcut: float) -> np.ndarray:
    """
    Apply a bandpass filter to the input signal.

    Parameters:
    - sig: np.ndarray
        The input signal.
    - fs: int
        The sampling frequency of the input signal.
    - lowcut: float
        The low cutoff frequency of the bandpass filter.
    - highcut: float
        The high cutoff frequency of the bandpass filter.

    Returns:
    - sig_filtered: np.ndarray
        The filtered signal using a Butterworth bandpass filter.
    """

    def butter_bandpass_filter(data, fs, lowcut, highcut, order=2):
        """
        Butterworth bandpass filter implementation.

        Parameters:
        - data: np.ndarray
            The input signal.
        - fs: int
            The sampling frequency of the input signal.
        - lowcut: float
            The low cutoff frequency of the bandpass filter.
        - highcut: float
            The high cutoff frequency of the bandpass filter.
        - order: int, optional
            The filter order (default is 2).

        Returns:
        - filtered_data: np.ndarray
            The filtered signal.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    # Apply bandpass filter to the input signal
    sig_filtered = butter_bandpass_filter(sig, fs, lowcut, highcut)

    return sig_filtered


        
def find_peaks(
        ppg: np.ndarray,
        sampling_rate: int,
        return_sig: bool = False):
    """
    Find peaks in PPG.

    Parameters:
    - ppg (np.ndarray): The input PPG signal.
    - sampling_rate (int): The sampling rate of the signal.
    - return_sig (bool, optional): If True, return the cleaned PPG signal along with the peak indices (default is False).

    Returns:
    - np.ndarray: An array containing the indices of the detected peaks in the PPG signal.
    - Optional: If return_sig is True, also returns the cleaned PPG signal.

    """

    # Clean the PPG signal and prepare it for peak detection
    ppg_cleaned = nk.ppg_clean(ppg, sampling_rate=sampling_rate)

    # Peak detection
    info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)
    peaks = info["PPG_Peaks"]

    # Return either just the peaks or both the cleaned signal and peaks
    if return_sig:
        return ppg_cleaned, peaks
    else:
        return peaks
        
        