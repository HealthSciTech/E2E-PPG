# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:44:39 2022

@author: mofeli
"""
# Import necessary libraries and modules


from PPG_SQA import ppg_sqa
from sklearn import preprocessing
import numpy as np
from scipy.signal import resample
import neurokit2 as nk
import torch



UPSAMPLING_RATE = 2
#Maximum reconstruction length (in seconds):
MAX_RECONSTRUCTION_LENGTH_SEC = 15


# Define a function to find peaks in a PPG segment
def find_peaks(ppg_segment, sampling_rate=20):
    # clean PPG signal and prepare it for peak detection
    ppg_cleaned = nk.ppg_clean(ppg_segment, sampling_rate=sampling_rate)
    # peak detection
    info  = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)
    peaks = info["PPG_Peaks"]
    return peaks


# Define a function to reconstruct a gap in the PPG signal using the GAN generator
def reconstruct(generator, device, ppg, gap, sampling_rate):
    
    # reconstruction of 5 sec signal in each iteration
    shifting_step = 5 
    len_rec = int(shifting_step*sampling_rate)
    
    ppg_clean = ppg
    reconstructed_gap = []
    gap_left = len(gap)
    while(len(reconstructed_gap) < len(gap)):
        ppg_clean = preprocessing.scale(ppg_clean)
        y = np.array([np.array(ppg_clean) for i in range(256)])
        ppg_test = torch.FloatTensor(y)
        rec_test,_ = generator(ppg_test.reshape(256,1,-1).to(device))
        rec_test = rec_test.cpu()
        re = rec_test.detach().numpy()


        rec_resampled = resample(re[0], len(re[0]) * UPSAMPLING_RATE)
        peaks_rec = find_peaks(rec_resampled)

        ppg_resampled = resample(ppg_clean, len(ppg_clean) * UPSAMPLING_RATE)
        peaks_ppg = find_peaks(ppg_resampled)

        ppg_and_rec = list(ppg_resampled[:peaks_ppg[-1]]) + list(rec_resampled[peaks_rec[0]:])
        
        len_ppg_and_rec = len(ppg_clean) + len(re[0])
        #Downsampling
        ppg_and_rec_down = resample(ppg_and_rec, len_ppg_and_rec)
        
        
        if gap_left < (len_rec):
            reconstructed_gap = reconstructed_gap + list(ppg_and_rec_down[len(ppg_clean):int((len(ppg_clean)+gap_left))])
        else:
            # select shifting_step sec of reconstructed signal
            reconstructed_gap = reconstructed_gap + list(ppg_and_rec_down[len(ppg_clean):(len(ppg_clean)+int(len_rec))])
            gap_left = gap_left - (len_rec)
            
        ppg_clean = ppg_and_rec_down[int(len_rec):len(ppg_clean)+int(len_rec)]
    
    
    return reconstructed_gap






# Define the main reconstruction function
def ppg_reconstruction(generator, device, sig, x_reliable, gaps, sampling_rate):
    sig_scaled= preprocessing.scale(sig)
    max_rec_length = int(MAX_RECONSTRUCTION_LENGTH_SEC*sampling_rate)
    reconstruction_flag = False
    for gap in gaps:
        if len(gap) <= max_rec_length:
            gap_start_idx = gap[0]
            # For reconstruction 15 sec clean preceding signal is needed
            if gap_start_idx >= max_rec_length:
                if set(range(gap_start_idx - max_rec_length, gap_start_idx)).issubset(x_reliable):
                    reconstructed_gap = reconstruct(generator, device, sig[gap_start_idx - max_rec_length : gap_start_idx], gap, sampling_rate)
                    reconstructed_gap_res = resample(reconstructed_gap, len(reconstructed_gap)*UPSAMPLING_RATE)
                    sig_before_gap_res = resample(sig_scaled[:gap_start_idx], len(sig_scaled[:gap_start_idx])*UPSAMPLING_RATE)
                    sig_after_gap_res = resample(sig_scaled[gap[-1]:], len(sig_scaled[gap[-1]:])*UPSAMPLING_RATE)

                    peaks_sig_before_gap = find_peaks(sig_before_gap_res)
    
                    if len(reconstructed_gap_res) >=80:
                        try:
                            peaks_gap_rec = find_peaks(reconstructed_gap_res)
                            if len(sig_after_gap_res) >= 80:
                                peaks_sig_after_gap = find_peaks(sig_after_gap_res)
    
                                sig_res = list(sig_before_gap_res[:peaks_sig_before_gap[-1]]) + \
                                list(reconstructed_gap_res[peaks_gap_rec[0]:peaks_gap_rec[-1]]) + \
                                list(sig_after_gap_res[peaks_sig_after_gap[0]:])
    
                            else:
                                sig_res = list(sig_before_gap_res[:peaks_sig_before_gap[-1]]) + \
                                list(reconstructed_gap_res[peaks_gap_rec[0]:peaks_gap_rec[-1]]) + \
                                list(sig_after_gap_res)
                        
                        except:
                            continue
                    else:
                        try:
                            if len(sig_after_gap_res) >= 80:
                                peaks_sig_after_gap = find_peaks(sig_after_gap_res)
    
                                sig_res = list(sig_before_gap_res[:peaks_sig_before_gap[-1]]) + \
                                list(reconstructed_gap_res) + \
                                list(sig_after_gap_res[peaks_sig_after_gap[0]:])
    
                            else:
                                sig_res = list(sig_before_gap_res[:peaks_sig_before_gap[-1]]) + \
                                list(reconstructed_gap_res) + \
                                list(sig_after_gap_res)
                        except:
                            continue
            
                    
                    sig_scaled= resample(sig_res, len(sig_scaled))
                    ppg_descaled = (sig_scaled*np.std(sig)) + np.mean(sig)
                    reconstruction_flag = True
                    x_reliable, gaps = ppg_sqa(ppg_descaled, sampling_rate)
                
    if reconstruction_flag == True:
        ppg_signal = ppg_descaled
    else:
        ppg_signal = sig
        
    return ppg_signal, x_reliable, gaps