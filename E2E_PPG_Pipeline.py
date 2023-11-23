# -*- coding: utf-8 -*-

import warnings
import pandas as pd
import numpy as np
from utils import bandpass_filter
from ppg_sqa import sqa
from ppg_reconstruction import reconstruction
from ppg_clean_extraction import clean_seg_extraction
from ppg_peak_detection import peak_detection
from ppg_hrv_extraction import hrv_extraction
warnings.filterwarnings("ignore")


def e2e_hrv_extraction(
        input_sig: np.ndarray,
        sampling_rate: int,
        window_length_sec: int
) -> pd.DataFrame:
    '''
    End-to-end HR and HRV extraction from an input PPG signal.
    
    Input parameters
        input_sig (np.ndarray): The input PPG signal.
        sampling_rate (int): The sampling rate of the input signal.
        window_length_sec (int): The desired window length for HR and HRV extraction in seconds.
        
    Returns
        hrv_data (pd.Dataframe): A DataFrame containing HRV parameters.

    '''
    # Apply bandpass filter if needed
    filtered_sig = bandpass_filter(
        sig=input_sig, fs=sampling_rate, lowcut=0.5, highcut=3)

    # Run signal quality assessment
    clean_indices, noisy_indices = sqa(
        sig=filtered_sig, sampling_rate=sampling_rate, filter_signal=False)

    # Run PPG reconstruction
    sig_reconstructed, clean_indices, noisy_indices = reconstruction(
        sig=filtered_sig,
        clean_indices=clean_indices,
        noisy_indices=noisy_indices,
        sampling_rate=sampling_rate,
        filter_signal=False)

    # Calculate the window length for HR and HRV extraction in terms of samples
    window_length = window_length_sec*sampling_rate

    # Scan clean parts of the signal and extract clean segments with the specified window length
    clean_segments = clean_seg_extraction(
        sig=sig_reconstructed,
        noisy_indices=noisy_indices,
        window_length=window_length)

    # Display results
    print("Analysis Results:")
    print("------------------")
    # Check if clean segments are found, if not, print a message
    if len(clean_segments) == 0:
        print('No clean ' + str(window_length_sec) + ' seconds segment was detected in the signal!')
    else:
        # Print the number of detected clean segments
        print(str(len(clean_segments)) + ' clean ' + str(window_length_sec) + ' seconds segments was detected in the signal!' )

        # Run PPG peak detection
        peaks, sampling_rate_new = peak_detection(clean_segments, sampling_rate)

        # Update window length based on the new sampling rate
        window_length_new = window_length_sec*sampling_rate_new

        # Perform HR and HRV extraction
        hrv_data = hrv_extraction(
            clean_segments=clean_segments,
            peaks=peaks,
            sampling_rate=sampling_rate_new,
            window_length=window_length_new)
        print("HR and HRV parameters:")
        print(hrv_data)
        print('Done!')

    return hrv_data
