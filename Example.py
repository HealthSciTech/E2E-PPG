# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:34:29 2023

@author: mofeli
"""

import pandas as pd
from  E2E_PPG_Pipeline import e2e_hrv_extraction

# Specify the input file and sample rate
file = 'data/201902020222_Data.csv'
sample_rate = 20


try:
    # Attempt to load data from the specified CSV file
    input_data = pd.read_csv(file, delim_whitespace=True, usecols=['timestamp','ppg'])
    ppg = input_data['ppg'].to_numpy()
    timestamp = input_data['timestamp'].to_numpy()
except:
    print('The input file can not be read, check the data type please!')
    
# Execute GAN.py script to import GAN model for PPG reconstruction
execfile('GAN.py')
reconstruction_model_parameters = [G, device]

# Set the window length for HR and HRV extraction in seconds
window_length_sec = 90

# Extract HRV features from the raw PPG signal using the E2E_PPG_Pipeline
hrv_data = e2e_hrv_extraction(ppg, timestamp, sample_rate, window_length_sec, reconstruction_model_parameters)

# Save HRV data to a CSV file
# filename = 'HRV_' + file
# hrv_data.to_csv(filename, index=False)


hrv_data.to_csv('HRV_data.csv', index=False)