# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras
import os
from utils import resample_signal

MODEL_PATH = "models"
PEAK_DETECTION_MODEL_DIR = "kazemi_peak_detection_model"

KAZEMI_MODEL_SAMPLING_FREQUENCYPLING_FREQUENCY = 100

def normalize(arr):
    """
    Normalize an array between (-1, 1).
    
    Args:
        arr (numpy.ndarray): An array of the signal

    Return:
        numpy.ndarray: Normalized array between (-1, 1)
    """
    return 1 * ((arr - arr.min()) / (arr.max() - arr.min()))


def split_signal(sig, rate, seconds, overlap, minlen):
    """
    Split a signal into segments.
    
    Args:
        sig (numpy.ndarray): Signal to be split
        rate (int): Sampling frequency of the signal
        seconds (int): Signal length in seconds
        overlap (int): Overlap in seconds
        minlen (int): Minimum length of the signal in seconds

    Return:
        sig_splits (numpy.ndarray): Segmentized signal
    """
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        if len(split) < int(minlen * rate):
            break

        sig_splits.append(split)

    sig_splits = np.asarray(sig_splits)
    sig_splits = sig_splits.reshape((sig_splits.shape[0], sig_splits.shape[1], 1))

    return sig_splits


def model_prediction(signal):
    """
    Load a pre-trained model and make prediction.
    
    Args:
        signal (numpy.ndarray): Signal for making predictions

    Return:
        prediction (numpy.ndarray): Model predictions
    """
    reconstructed_model = keras.models.load_model(os.path.join(MODEL_PATH, PEAK_DETECTION_MODEL_DIR))
    prediction = reconstructed_model.predict(signal)
    return prediction

def Wrapper_function(prediction, raw_signal):
    """
    Args:
        prediction (numpy.ndarray): Model predictions
        raw_signal (numpy.ndarray): Original raw signal

    Return:
        final_indeces (numpy.ndarray): Final indices of identified peaks
    """
    ## Normalizing the signal for post processing
    test = normalize(prediction)
    j = 0
    indeces = []
    ## Finding the peaks index
    while (j < len(test)-3):
        ## adjustinf the threshhold for the peak detection
        if test[j]>= 0.70:
            
            if j< len(test)-15:
                
                period = test[j:j+15]
                period_X = raw_signal[j:j+15]
                index = np.asarray(np.where(period==np.max(period)))
                if len(index[0])>1:
                    length = len(index[0])
                    index = index[0].tolist()
                    max_index = np.asarray(
                        np.where(period_X==np.max(period_X[index])))
                    indeces.append(int(max_index[0][0]+j)) 
                else:
                    max_index = np.asarray(
                        np.where(period_X==np.max(period_X[index])))
                    indeces.append(int(index[0]+j))
                j = j+15
            else:
                period = test[j:j+7]
                period_X = raw_signal[j:j+7]
                index = np.asarray(np.where(
                    period==np.max(period)))
                if (len(index[0])>1):
                    length = len(index[0])
                    index = index[0].tolist()
                    max_index = np.asarray(np.where(period_X==np.max(period_X[index])))
                    indeces.append(int(max_index[0][0]+j))
                j = j+7
        else:
            j +=1

    e = 0
    while (e<len(indeces)-1):
        if (indeces[e+1]-indeces[e]<35):
            if raw_signal[indeces[e+1]] < raw_signal[indeces[e]]:
                del (indeces[e+1])
            else:
                del (indeces[e])
            e = e

        else:
            e += 1

    del_index = []
    for y in range(len(indeces)):
        if raw_signal[indeces[y]]<=0:
            del_index.append(indeces[y])
      
    final_indeces = np.array([q for q in indeces if q not in del_index])

    return final_indeces


def ppg_peaks(signal, sampling_freq, seconds, overlap, minlen):
    """
    Main function to detect peaks in PPG signals using the trained model.
    
    Args:
        signal (numpy.ndarray): PPG signal
        sampling_freq (int): Sampling frequency of the signal
        seconds (int): Signal length in seconds
        overlap (int): Overlap in seconds
        minlen (int): Minimum length of the signal in seconds

    Return:
        peak_indexes (list): A list containing peak indexes 
        
    Reference:
        Kazemi, K., Laitala, J., Azimi, I., Liljeberg, P., & Rahmani, A. M. (2022). 
        Robust ppg peak detection using dilated convolutional neural networks. Sensors, 22(16), 6054.
    """
    # Upsample the signal if the sampling frequency is not 100 Hz
    
    resampling_flag = False
    # Check if resampling is needed and perform resampling if necessary
    if sampling_freq != KAZEMI_MODEL_SAMPLING_FREQUENCYPLING_FREQUENCY:
        signal = resample_signal(
            sig=signal, fs_origin=sampling_freq, fs_target=KAZEMI_MODEL_SAMPLING_FREQUENCYPLING_FREQUENCY)
        resampling_flag = True
        resampling_rate = sampling_freq/KAZEMI_MODEL_SAMPLING_FREQUENCYPLING_FREQUENCY
        sampling_freq = KAZEMI_MODEL_SAMPLING_FREQUENCYPLING_FREQUENCY
    
    # Split the signal into segments
    segmentized_signal = split_signal(signal, sampling_freq, seconds, overlap, minlen)

    # Make predictions using the pre-trained model and identify peaks
    prediction = model_prediction(segmentized_signal)
    indices = []

    # Process each 15-seconds segmentized signal
    for i in range(len(segmentized_signal)):
        # Call the wrapper function
        peak_index = Wrapper_function(prediction[i], segmentized_signal[i])
        peak_index = [item + sampling_freq * i * seconds for item in peak_index]
        indices.append(peak_index)

    peak_indexes = [item for sublist in indices for item in sublist]
    
    # If resampling performed, update indices according to the original sampling rate
    if resampling_flag:
        peak_indexes = [int(peak * resampling_rate) for peak in peak_indexes]
    
    return peak_indexes
