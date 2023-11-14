# -*- coding: utf-8 -*-
"""
PPG_Peak Detection
Author:  Kianoosh Kazemi
"""



from scipy.signal import butter, filtfilt
from scipy import stats
import neurokit2 as nk
import numpy as np
from wfdb import processing
from tensorflow import keras
from scipy.signal import resample

def normalize(arr):
  '''
  Parameters: an array of signal

  Return: normalized array between (-1,1)

  '''
  return 1 * ((arr - arr.min()) / (arr.max() - arr.min()))

# def butter_filtering(sig,fs,fc,order,btype): 
#   '''
#   Parameters: signal, sampling frequency, cutoff frequencies, order of the filter, filter type (e.g., bandpass)

#   Returns: filtered signal
#   '''

#   w = np.array(fc)/(fs/2)
#   b, a = butter(order, w, btype =btype, analog=False)
#   filtered = filtfilt(b, a, sig)
#   return(filtered)

def upsample(signal, org_fs):
  '''
  Parameters: signal, old sampling frequency

  Return: upsamle signal

  '''
  ## Our Dilated CNN is trained with 100 Hz sampling rate
  model_fs = 100

  resampled_sig , resampled_t = processing.resample_sig(signal, org_fs, model_fs)

  return resampled_sig

def splitSignal(sig, rate , seconds, overlap, minlen):
  '''
  Parameters: signal, sampling frequency, signal length in seconds, overlap in seconds, minimum length of the signal in seconds
  
  Return: Segmentized signal

  '''
  
  # Split signal with overlap
  sig_splits = []
  for i in range(0, len(sig), int((seconds - overlap) * rate)):
      split = sig[i:i + int(seconds * rate)]

      # End of signal?
      if len(split) < int(minlen * rate):
          break
      
      sig_splits.append(split)

  sig_splits = np.asarray(sig_splits) 
  sig_splits = sig_splits.reshape((sig_splits.shape[0], sig_splits.shape[1], 1))

  return sig_splits

def model_prediction(signal):

  reconstructed_model = keras.models.load_model('trained_model')

  prediction = reconstructed_model.predict(signal)

  return prediction

def Wrapper_function(prediction, raw_signal):
  

  ## Normalizing the signal for post processing
  test = normalize(prediction)
  j = 0
  indeces = []

  while (j < len(test)-3):
      
      if test[j]>= 0.70:
          
          if j< len(test)-15:
              
              period = test[j:j+15]
              period_X = raw_signal[j:j+15]
              index = np.asarray(np.where(period==np.max(period)))
              if len(index[0])>1:
                  length = len(index[0])
                  index = index[0].tolist()
                  max_index = np.asarray(np.where(period_X==np.max(period_X[index])))
                  indeces.append(int(max_index[0][0]+j))
                  
                  
              else:
                  max_index = np.asarray(np.where(period_X==np.max(period_X[index])))
                  indeces.append(int(index[0]+j))
                  

              j = j+15
          else:
              period = test[j:j+7]
              period_X = raw_signal[j:j+7]
              index = np.asarray(np.where(period==np.max(period)))
              
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

def Neurokit_peak(signal, sampling_rate):
  signals, info = nk.ppg_process(signal, sampling_rate=sampling_rate)
  peaks = info['PPG_Peaks'] 

  return signals, peaks

def PPG_Peak(signal, sampling_freq, seconds, overlap, minlen):
    '''
    Parameters: signal, sampling frequency, cutoff frequencies, order of the filter, filter type (e.g., bandpass), signal length in seconds, overlap in seconds, minimum length of the signal in seconds
    
    Return: Segmentized signal
    
    '''
    ## Calling the upsampling method
    if sampling_freq != 100:
        resampling_rate = 100/sampling_freq
        signal = resample(signal, int(len(signal)*resampling_rate))
        sampling_freq = 100
    # upsampled_signal = upsample(signal, sampling_freq) 
    
    ## Calling the filtering method
    # filtered_signal = butter_filtering(upsampled_signal, sampling_freq, fc, order, btype)
    
    ## Calling the splitting method
    segmentized_signal = splitSignal(signal, sampling_freq, seconds, overlap, minlen)
    
    ## Calling the model and wrapper function to detect the signals peak location
    
    prediction = model_prediction(segmentized_signal)
    indeces = []
    ## Going through all the 15-seconds segmentized signal
    for i in range(len(segmentized_signal)):
    
      ## Calling Wrapper function
      peak_index = Wrapper_function(prediction[i],segmentized_signal[i])
      peak_index = [item + sampling_freq*i*seconds for item in peak_index]
      ## Appending all the indeces
      indeces.append(peak_index)
    
    peak_indexes = [item for sublist in peak_index for item in sublist]

    return peak_indexes
