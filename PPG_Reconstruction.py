# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:44:39 2022

@author: mofeli
"""
# Import necessary libraries and modules


from PPG_SQA import PPG_SQA
from sklearn import preprocessing
import numpy as np
from scipy.signal import resample
import neurokit2 as nk
import torch



    


# Define a function to find peaks in a PPG segment
def find_peaks(ppg_segment, sampling_rate=20):
    # clean PPG signal and prepare it for peak detection
    ppg_cleaned = nk.ppg_clean(ppg_segment, sampling_rate=sampling_rate)
    # peak detection
    info  = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=sampling_rate)
    peaks = info["PPG_Peaks"]
    return peaks


# Define a function to reconstruct a gap in the PPG signal using the GAN generator
def ppg_reconstruct(generator, device, ppg, gap, sample_rate):
    
    shifting_step = 5 
    # we reconstruct 5 sec ppg
    len_rec = 5*sample_rate
    
    ppg_clean = ppg
    gap_reconstructed = []
    gap_left = len(gap)
    while(len(gap_reconstructed) < len(gap)):
        ppg_clean = preprocessing.scale(ppg_clean)
        y = np.array([np.array(ppg_clean) for i in range(256)])
        ppg_test = torch.FloatTensor(y)
        rec_test,_ = generator(ppg_test.reshape(256,1,-1).to(device))
        rec_test = rec_test.cpu()
        re = rec_test.detach().numpy()

        upsampling_rate=2
        rec_resampled = resample(re[0], len(re[0]) * upsampling_rate)
        peaks_rec = find_peaks(rec_resampled)

        ppg_resampled = resample(ppg_clean, len(ppg_clean) * upsampling_rate)
        peaks_ppg = find_peaks(ppg_resampled)

        ppg_and_rec = list(ppg_resampled[:peaks_ppg[-1]]) + list(rec_resampled[peaks_rec[0]:])
        
        # len(ppg_clean) + len(recunstructed) should be 300+100= 400
        len_ppg_and_rec = len(ppg_clean) + len(re[0])
        #Downsampling
        ppg_and_rec_down = resample(ppg_and_rec, len_ppg_and_rec)
        
        
        if gap_left < (shifting_step*sample_rate):
            gap_reconstructed = gap_reconstructed + list(ppg_and_rec_down[len(ppg_clean):int((len(ppg_clean)+gap_left))])
        else:
            # select shifting_step sec of reconstructed signal
            gap_reconstructed = gap_reconstructed + list(ppg_and_rec_down[len(ppg_clean):(len(ppg_clean)+int(shifting_step*sample_rate))])
            gap_left = gap_left - (shifting_step*sample_rate)
            
        ppg_clean = ppg_and_rec_down[int(shifting_step*sample_rate):len(ppg_clean)+int(shifting_step*sample_rate)]
    
    
    return gap_reconstructed






# Define the main reconstruction function
def ppg_reconstruction(generator, device, ppg_filt, x_reliable, gaps, sample_rate):
    upsampling_rate=2
    ppg_original = preprocessing.scale(ppg_filt)
    reconstructed_flag = False
    for gap in gaps:
        if len(gap) <= (15*sample_rate):
            if gap[0] >= 300:
                if set(range(gap[0]-300,gap[0])).issubset(x_reliable):
                    # print(type(ppg_filt))
                    # print(type(gap))
                    # print(len(gap))
                    gap_reconstructed = ppg_reconstruct(generator, device, ppg_filt[gap[0]-300:gap[0]],gap, sample_rate)
                    # print('reconstruction applied')
                    gap_reconstructed_res = resample(gap_reconstructed, len(gap_reconstructed)*upsampling_rate)
                    ppg_original_before_gap_res = resample(ppg_original[:gap[0]], len(ppg_original[:gap[0]])*upsampling_rate)
                    ppg_original_after_gap_res = resample(ppg_original[gap[-1]:], len(ppg_original[gap[-1]:])*upsampling_rate)

                    peaks_ppg_original_before_gap = find_peaks(ppg_original_before_gap_res)
    
                    if len(gap_reconstructed_res) >=80:
                        try:
                            peaks_gap_rec = find_peaks(gap_reconstructed_res)
                            if len(ppg_original_after_gap_res) >= 80:
                                peaks_ppg_original_after_gap = find_peaks(ppg_original_after_gap_res)
    
                                ppg_original_res = list(ppg_original_before_gap_res[:peaks_ppg_original_before_gap[-1]]) + \
                                list(gap_reconstructed_res[peaks_gap_rec[0]:peaks_gap_rec[-1]]) + \
                                list(ppg_original_after_gap_res[peaks_ppg_original_after_gap[0]:])
    
                            else:
                                ppg_original_res = list(ppg_original_before_gap_res[:peaks_ppg_original_before_gap[-1]]) + \
                                list(gap_reconstructed_res[peaks_gap_rec[0]:peaks_gap_rec[-1]]) + \
                                list(ppg_original_after_gap_res)
                        
                        except:
                            continue
                    else:
                        try:
                            if len(ppg_original_after_gap_res) >= 80:
                                peaks_ppg_original_after_gap = find_peaks(ppg_original_after_gap_res)
    
                                ppg_original_res = list(ppg_original_before_gap_res[:peaks_ppg_original_before_gap[-1]]) + \
                                list(gap_reconstructed_res) + \
                                list(ppg_original_after_gap_res[peaks_ppg_original_after_gap[0]:])
    
                            else:
                                ppg_original_res = list(ppg_original_before_gap_res[:peaks_ppg_original_before_gap[-1]]) + \
                                list(gap_reconstructed_res) + \
                                list(ppg_original_after_gap_res)
                        except:
                            continue
            
                    
                    ppg_original= resample(ppg_original_res, len(ppg_original))
                    ppg_descaled = (ppg_original*np.std(ppg_filt)) + np.mean(ppg_filt)
                    reconstructed_flag = True
                    x_reliable, gaps = PPG_SQA(ppg_descaled, sample_rate, doPlot=False)
                
    if reconstructed_flag == True:
        ppg_signal = ppg_descaled
    else:
        ppg_signal = ppg_filt
        
    return ppg_signal, x_reliable, gaps