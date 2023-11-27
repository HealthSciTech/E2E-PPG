# -*- coding: utf-8 -*-

import os
import warnings
from e2e_ppg_pipeline import e2e_hrv_extraction
from utils import get_data
warnings.filterwarnings("ignore")


# Import a sample data
file_name = "201902020222_Data.csv"
sampling_frequency = 20
input_sig = get_data(file_name=file_name)

# Set the window length for HR and HRV extraction in seconds
window_length_sec = 90

# Extract HRV parameters from the input PPG signal
hrv_data = e2e_hrv_extraction(
    input_sig=input_sig,
    sampling_rate=sampling_frequency,
    window_length_sec=window_length_sec)

# Output file name
output_file_name = "_".join(['HRV_', file_name])

# Save HRV data to a CSV file
hrv_data.to_csv(os.path.join('data', output_file_name), index=False)
