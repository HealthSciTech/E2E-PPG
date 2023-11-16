
# -*- coding: utf-8 -*-
'''
Miscellaneous functions

'''

import numpy as np
import pandas as pd
from typing import List
import os


def get_data(
        local_directory: str,
        file_name: str,
        usecols: List[str] = ['timestamp','ppg'],
) -> pd.DataFrame:

    try:
        # Attempt to load data from the specified CSV file
        input_data = pd.read_csv(
            os.path.join(local_directory, file_name),
            delim_whitespace=True,
            usecols=usecols)
        ppg = input_data['ppg'].to_numpy()
        timestamp = input_data['timestamp'].to_numpy()
    except:
        print('The input file can not be read, check the data type please!')


def normalize_data(sig: np.ndarray) -> np.ndarray:
    """
    Normalize the input signal between zero and one
    
    Parameters:
        sig: PPG signal.
    
    Returns:
        Normalized signal
    """
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))