# -*- coding: utf-8 -*-

import os
from e2e_ppg_pipeline import e2e_hrv_extraction
from utils import get_data
import warnings
warnings.filterwarnings("ignore")


# Import a sample data
FILE_NAME = "201902020222_Data.csv"
SAMPLING_FREQUENCY = 20
input_sig = get_data(file_name=FILE_NAME)
    
# Set the window length for HR and HRV extraction in seconds
window_length_sec = 90

# Extract HRV parameters from the input PPG signal
hrv_data = e2e_hrv_extraction(input_sig=input_sig, sampling_rate=SAMPLING_FREQUENCY, window_length_sec=window_length_sec)

# Output file name
OUTPUT_FILENAME = 'HRV_' + FILE_NAME

# Save HRV data to a CSV file
hrv_data.to_csv(os.path.join('data', OUTPUT_FILENAME), index=False)
