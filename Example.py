# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:34:29 2023

@author: mofeli
"""

import pandas as pd
from  E2E_PPG_Pipeline import HRV_Extraction

# Specify the input file and sample rate
file = '201902020222_Data.csv'
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

# Set the window length for HR and HRV extraction in min
window_length_min = 1.5

# Extract HRV features from the raw PPG signal using the E2E_PPG_Pipeline
hrv_data = HRV_Extraction(ppg, timestamp, sample_rate, window_length_min, reconstruction_model_parameters)

# Save HRV data to a CSV file
filename = 'HRV_' + file
hrv_data.to_csv(filename, index=False)
