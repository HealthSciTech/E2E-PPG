
# -*- coding: utf-8 -*-
'''
Miscellaneous functions

'''

import numpy as np
import pandas as pd
from typing import List, Tuple
import os


def get_data(
        file_name: str,
        local_directory: str = "data",
        usecols: List[str] = ['ppg'],
) -> np.ndarray:
    """
    Import data
    
    Input parameters:
        file_name: Name of the input file
        local_directory: Data directory
        usecols: The columns to read from the input file
    
    Returns:
        sig: the input signal (e.g., PPG)
    """
    try:
        # Construct the file path
        file_path = os.path.join(local_directory, file_name)
        # Load data from the specified CSV file
        input_data = pd.read_csv(
            file_path,
            delim_whitespace=True,
            usecols=usecols)
        # Extract signal
        sig = input_data[usecols[0]].values
        return sig
    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except pd.errors.EmptyDataError:
        print(f"Empty data in file: {file_name}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # Return None in case of an error
    return None



def normalize_data(sig: np.ndarray) -> np.ndarray:
    """
    Normalize the input signal between zero and one
    
    Parameters:
        sig: PPG signal.
    
    Returns:
        Normalized signal
    """
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))