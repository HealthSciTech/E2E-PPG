# -*- coding: utf-8 -*-

import neurokit2 as nk
import heartpy as hp
from heartpy.datautils import rolling_mean
import numpy as np
from scipy import signal
from kazemi_peak_detection import ppg_peaks
from ppg_sqa import sqa
from ppg_reconstruction import reconstruction
from ppg_clean_extraction import clean_seg_extraction
from utils import normalize_data, get_data

import warnings
warnings.filterwarnings("ignore")


def peak_detection(
        clean_segments: list, 
        sampling_rate: int, 
        method: str ='kazemi') -> list:
    '''
    Detect peaks in clean PPG segments using specified peak detection method.
    
    Args:
        clean_segments (list): List of clean PPG segments with the specified window length and their starting index.
        sampling_rate: Sampling rate of the PPG signal.
        method (str): Peak detection method. Valid inputs: 'nk', 'kazemi', and  'heartpy'. The default is 'kazemi'. (optional)

    Return:
        total_peaks (list): List of lists, each containing the detected peaks for a corresponding clean segment.
        
    Refernces:
        Kazemi method: Kazemi, K., Laitala, J., Azimi, I., Liljeberg, P., & Rahmani, A. M. (2022). 
            Robust ppg peak detection using dilated convolutional neural networks. Sensors, 22(16), 6054.
        Neurokit method: Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., ... & Chen, S. A. (2021).
            NeuroKit2: A Python toolbox for neurophysiological signal processing. Behavior research methods, 1-8.
        HeartPY method: Van Gent, P., Farah, H., Nes, N., & van Arem, B. (2018, June). 
            Heart rate analysis for human factors: Development and validation of an open source toolkit for noisy naturalistic heart rate data. 
            In Proceedings of the 6th HUMANIST Conference (pp. 173-178). 
    '''
    # Initialize a list to store total peaks
    total_peaks = []
    
    # Check the deisred peak detection method
    if method == 'nk':
        # Neurokit method
        upsampling_rate = 2
        sampling_rate_new = sampling_rate * upsampling_rate
        
        for i in range(len(clean_segments)):
            # Normalize PPG signal
            ppg_normed = normalize_data(clean_segments[i][1])
            
            # Upsampling the signal
            resampled = signal.resample(ppg_normed, len(ppg_normed) * upsampling_rate)
            
            # Perform peak detection 
            ppg_cleaned = nk.ppg_clean(resampled, sampling_rate=sampling_rate_new)
            info  = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate_new)
            peaks = info["PPG_Peaks"]
            
            # Update peak indices according to the original sampling rate
            peaks = (peaks // upsampling_rate).astype(int)
            
            # Add peaks of the current segment to the total peaks
            total_peaks.append(peaks)
            
        # Return total peaks
        return total_peaks
    
    elif method == 'kazemi':
        # Kazemi method
        for i in range(len(clean_segments)):
            # Perform peak detection
            peaks = ppg_peaks(np.asarray(clean_segments[i][1]), sampling_rate, seconds = 15, overlap = 0, minlen = 15)
            
            # Add peaks of the current segment to the total peaks
            total_peaks.append(peaks)
            
        # Return total peaks 
        return total_peaks

    elif method == 'heartpy':
        # HeartPy method
        for i in range(len(clean_segments)):
            # Perform peak detection
            rol_mean = rolling_mean(clean_segments[i][1], windowsize = 0.75, sample_rate = sampling_rate)
            wd = hp.peakdetection.detect_peaks(np.array(clean_segments[i][1]), rol_mean, ma_perc = 20, sample_rate = sampling_rate)
            peaks = wd['peaklist']
            
            # Add peaks of the current segment to the total peaks
            total_peaks.append(peaks)
            
        # Return total peaks 
        return total_peaks
        
    else:
        print("Invalid method. Please choose from 'neurokit', 'kazemi', or 'heartpy'")
        return None




if __name__ == "__main__":
    # Import a sample data
    file_name = "201902020222_Data.csv"
    input_sampling_rate = 20
    input_sig = get_data(file_name=file_name)

    # Define a window length for clean segments extraction (in seconds)
    window_length_sec = 60
    
    # Run PPG signal quality assessment.
    clean_ind, noisy_ind = sqa(sig=input_sig, sampling_rate=input_sampling_rate)
    
    # Run PPG reconstruction
    reconstructed_signal, clean_ind, noisy_ind = reconstruction(
        sig=input_sig,
        clean_indices=clean_ind,
        noisy_indices=noisy_ind,
        sampling_rate=input_sampling_rate)


    # Calculate the window length in terms of samples
    window_length = window_length_sec*input_sampling_rate
    
    # Scan clean parts of the signal and extract clean segments with the specified window length
    clean_segments = clean_seg_extraction(sig=reconstructed_signal, noisy_indices=noisy_ind, window_length=window_length)

    # Display results
    print("Analysis Results:")
    print("------------------")
    # Check if clean segments are found, if not, print a message
    if len(clean_segments) == 0:
        print('No clean ' + str(window_length_sec) + ' seconds segment was detected in the signal!')
    else:
        # Print the number of detected clean segments
        print(str(len(clean_segments)) + ' clean ' + str(window_length_sec) + ' seconds segments was detected in the signal!' )

        # Run PPG Peak detection
        peaks = peak_detection(
            clean_segments=clean_segments, sampling_rate=input_sampling_rate)
        print("Number of detected peaks in each segment:")
        for pks in peaks:
            print(len(pks))
