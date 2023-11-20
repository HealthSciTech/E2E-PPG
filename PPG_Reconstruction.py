# -*- coding: utf-8 -*-

from ppg_sqa import sqa
from sklearn import preprocessing
import numpy as np
from scipy.signal import resample
import torch
from utils import find_peaks, check_and_resample, get_data, bandpass_filter
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

UPSAMPLING_RATE = 2
#Maximum reconstruction length (in seconds):
MAX_RECONSTRUCTION_LENGTH_SEC = 15


def gan_rec(
        ppg_clean: np.ndarray,
        noise: list,
        sampling_rate: int,
        generator, device) -> list:
    
    '''
    Reconstruct noise in the PPG signal using a Generative Adversarial Network (GAN) generator.
    
    Input parameters:
        ppg_clean (np.ndarray): Preceding clean signal to be used for noise reconstruction.
        noise (list): List of noise indices.
        sampling_rate (int): Sampling rate of the PPG signal.
        generator: GAN generator for noise reconstruction.
        device: Device on which to run the generator.
        
    Returns:
        reconstructed_noise: The reconstructed noise.
        
        
    This function iteratively reconstructs noise in the PPG signal using a GAN generator.
    The clean signal and reconstructed noise are concatenated, and the process is repeated
    until the entire noise sequence is reconstructed.
    '''
    
    # Parameters for reconstruction
    shifting_step = 5 
    len_rec = int(shifting_step*sampling_rate)
    reconstructed_noise = []
    remaining_noise = len(noise)
    
    # Iteratively reconstruct noise
    while(len(reconstructed_noise) < len(noise)):
        # Preprocess and prepare the clean signal for reconstruction
        ppg_clean = preprocessing.scale(ppg_clean)
        y = np.array([np.array(ppg_clean) for i in range(256)])
        ppg_test = torch.FloatTensor(y)
        
        # Generate reconstructed noise using the GAN generator
        rec_test,_ = generator(ppg_test.reshape(256,1,-1).to(device))
        rec_test = rec_test.cpu()
        noise_rec = rec_test.detach().numpy()[0]
        
        # Find peaks in the reconstructed noise
        noise_rec_res = resample(noise_rec, int(len(noise_rec) * UPSAMPLING_RATE))
        peaks_rec = find_peaks(noise_rec_res, int(sampling_rate * UPSAMPLING_RATE))
        
        # Find peaks in the clean signal
        ppg_resampled = resample(ppg_clean, int(len(ppg_clean) * UPSAMPLING_RATE))
        peaks_ppg = find_peaks(ppg_resampled, int(sampling_rate * UPSAMPLING_RATE))

        # Concatenate reconstructed noise with preceding clean signal
        ppg_and_rec = list(ppg_resampled[:peaks_ppg[-1]]) + list(noise_rec_res[peaks_rec[0]:])
        
        len_ppg_and_rec = len(ppg_clean) + len(noise_rec)
        
        # Resample to the original length of the clean signal and reconstructed noise
        ppg_and_rec_down = resample(ppg_and_rec, len_ppg_and_rec)
        
        # Update reconstructed noise based on the remaining noise
        if remaining_noise < (len_rec):
            reconstructed_noise = reconstructed_noise + list(ppg_and_rec_down[len(ppg_clean):int((len(ppg_clean)+remaining_noise))])
        else:
            # Select the reconstructed part of the signal 
            reconstructed_noise = reconstructed_noise + list(ppg_and_rec_down[len(ppg_clean):(len(ppg_clean)+int(len_rec))])
            # Update remaining noise value
            remaining_noise = remaining_noise - (len_rec)
        
        # Select the next segment of the clean signal for the next iteration (next reconstruction)
        ppg_clean = ppg_and_rec_down[int(len_rec):len(ppg_clean)+int(len_rec)]
    
    return reconstructed_noise


def reconstruction(
        sig: np.ndarray,
        clean_indices: list,
        noisy_indices:list,
        sampling_rate: int,
        generator,device) -> Tuple[np.ndarray, list, list]:
    
    '''
    Main function for PPG reconstruction.

    Input parameters:
    - sig (np.ndarray): Original PPG signal.
    - clean_indices (list): List of indices representing clean parts.
    - noisy_indices (list): List of indices representing noisy parts.
    - sampling_rate (int): Sampling rate of the signal.
    - generator: GAN generator for noise reconstruction.
    - device: Device on which to run the generator.

    Returns:
    - ppg_signal (np.ndarray): Reconstructed PPG signal (if reconstruction is applied; otherwise, returns the original signal).
    - clean_indices (list): Updated indices of clean parts (if reconstruction is applied; otherwise, returns the original indices of clean parts).
    - noisy_indices (list): Updated indices of noisy parts (if reconstruction is applied;, otherwise, returns the original indices of noisy parts).

    '''
    # Scale the original PPG signal for further processing
    sig_scaled= preprocessing.scale(sig)
    
    # Maximum length for reconstruction
    max_rec_length = int(MAX_RECONSTRUCTION_LENGTH_SEC*sampling_rate)
    
    # Flag to indicate if reconstruction has occurred
    reconstruction_flag = False
    
    # Iterate over noisy parts for reconstruction
    for noise in noisy_indices:
        if len(noise) <= max_rec_length:
            noise_start_idx = noise[0]
            # Check if there is sufficient preceding clean signal for reconstruction
            if noise_start_idx >= max_rec_length:
                # Check if the preceding signal is clean
                if set(range(noise_start_idx - max_rec_length, noise_start_idx)).issubset(clean_indices):
                    # Perform noise reconstruction for the current noise
                    reconstructed_noise = gan_rec(sig[noise_start_idx - max_rec_length : noise_start_idx], noise, sampling_rate, generator, device)
                    
                    # Upsample the reconstructed noise
                    reconstructed_noise_res = resample(reconstructed_noise, int(len(reconstructed_noise)*UPSAMPLING_RATE))
                    
                    # Upsample the clean signal before the noise
                    sig_before_noise_res = resample(sig_scaled[:noise_start_idx], int(len(sig_scaled[:noise_start_idx])*UPSAMPLING_RATE))
                    
                    # Upsample the clean signal after the noise
                    sig_after_noise_res = resample(sig_scaled[noise[-1]:], int(len(sig_scaled[noise[-1]:])*UPSAMPLING_RATE))

                    # Find peaks in the clean signal before the noise        
                    peaks_sig_before_noise = find_peaks(sig_before_noise_res, int(sampling_rate*UPSAMPLING_RATE))
                    
                    # Check if the reconstructed noise is long enough (considering a threshold of 2 seconds)
                    if len(reconstructed_noise_res) >= 2*sampling_rate*UPSAMPLING_RATE:
                        try:
                            # Find peaks in the reconstructed noise
                            peaks_noise_rec = find_peaks(reconstructed_noise_res, int(sampling_rate*UPSAMPLING_RATE))
                            
                            # Check if the clean signal after the noise is long enough (considering a threshold of 2 seconds)
                            if len(sig_after_noise_res) >= 2*sampling_rate*UPSAMPLING_RATE:
                                # Find peaks in the clean signal after the noise 
                                peaks_sig_after_noise = find_peaks(sig_after_noise_res, int(sampling_rate*UPSAMPLING_RATE))
                                
                                # Merge the reconstructed noise with the clean signal
                                sig_res = list(sig_before_noise_res[:peaks_sig_before_noise[-1]]) + \
                                list(reconstructed_noise_res[peaks_noise_rec[0]:peaks_noise_rec[-1]]) + \
                                list(sig_after_noise_res[peaks_sig_after_noise[0]:])
                                
                            # If the clean signal after the noise is too short, there is no need for peak detection
                            else:
                                # Merge the reconstructed noise with the clean signal
                                sig_res = list(sig_before_noise_res[:peaks_sig_before_noise[-1]]) + \
                                list(reconstructed_noise_res[peaks_noise_rec[0]:peaks_noise_rec[-1]]) + \
                                list(sig_after_noise_res)
                        except:
                            continue
        
                    else:
                        try:
                            # Check if the clean signal after the noise is long enough (considering a threshold of 2 seconds)
                            if len(sig_after_noise_res) >= 2*sampling_rate*UPSAMPLING_RATE:
                                # Find peaks in the clean signal after the noise 
                                peaks_sig_after_noise = find_peaks(sig_after_noise_res, int(sampling_rate*UPSAMPLING_RATE))
                                
                                # Merge the reconstructed noise with the clean signal
                                sig_res = list(sig_before_noise_res[:peaks_sig_before_noise[-1]]) + \
                                list(reconstructed_noise_res) + \
                                list(sig_after_noise_res[peaks_sig_after_noise[0]:])
                                
                            # If the clean signal after the noise is too short, there is no need for peak detection
                            else:
                                # Merge the reconstructed noise with the clean signal
                                sig_res = list(sig_before_noise_res[:peaks_sig_before_noise[-1]]) + \
                                list(reconstructed_noise_res) + \
                                list(sig_after_noise_res)
                        except:
                            continue
            
                    # Resample the reconstructed signal to the original length of the signal
                    sig_scaled= resample(sig_res, len(sig_scaled))
                    
                    # Descale the reconstructed signal
                    ppg_descaled = (sig_scaled*np.std(sig)) + np.mean(sig)
                    
                    # Set the reconstruction flag to True
                    reconstruction_flag = True
                    
                    # Perform the signal quality assessment to ensure that the reconstructed signal is not distorted
                    clean_indices, noisy_indices = sqa(sig=ppg_descaled, sampling_rate=sampling_rate)
    
    # Check if there was a reconstruction
    if reconstruction_flag == True:
        ppg_signal = ppg_descaled
    else:
        ppg_signal = sig
        
    # Return the reconstructed or original PPG signal, along with updated indices
    return ppg_signal, clean_indices, noisy_indices


if __name__ == "__main__":
    # Import a sample data
    FILE_NAME = "201902020222_Data.csv"
    SAMPLING_FREQUENCY = 20
    input_sig = get_data(file_name=FILE_NAME)
    
    # Check if resampling is needed and perform resampling if necessary
    input_sig, sampling_rate = check_and_resample(sig=input_sig, fs=SAMPLING_FREQUENCY)
    
    # Bandpass filter parameters
    lowcut = 0.5  # Lower cutoff frequency in Hz
    highcut = 3  # Upper cutoff frequency in Hz
    
    # Apply bandpass filter
    filtered_sig = bandpass_filter(sig=input_sig, fs=sampling_rate, lowcut=lowcut, highcut=highcut)

    # Run PPG signal quality assessment.
    clean_indices, noisy_indices = sqa(sig=filtered_sig, sampling_rate=sampling_rate)
    
    execfile('GAN.py')
    reconstruction_model_parameters = [G, device]
    
    # Run PPG reconstruction
    ppg_signal, clean_indices, noisy_indices = reconstruction(sig=filtered_sig, clean_indices=clean_indices, noisy_indices=noisy_indices, sampling_rate=sampling_rate, generator=G, device=device)
    
    # Display results
    print("Analysis Results:")
    print("------------------")
    print(f"Length of the clean signal after reconstruction (in seconds): {len(clean_indices)/SAMPLING_FREQUENCY:.2f}")
    print(f"Number of noisy parts in the signal after reconstruction: {len(noisy_indices)}")
    
    if len(noisy_indices) > 0:
        print("Length of each noise in the signal after reconstruction (in seconds):")
        for noise in noisy_indices:
            print(f"   - {len(noise)/SAMPLING_FREQUENCY:.2f}")