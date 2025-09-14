# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ACTL3143)
#     language: python
#     name: actl3143
# ---

# %%
import os
import json
import pickle
from datetime import datetime

# All output files will be stored in the /data folder (relative to the notebook location)
SAVE_DIR = os.path.join(os.getcwd(), "data")  # You may also rename it to 'output' or 'artifacts'

def get_data_path(filename):
    """
    Get the full path to load a data file (e.g., .csv, .json, .pkl).
    """
    base = SAVE_DIR
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)

def get_save_path(filename):
    """
    Get the full path for saving a file (supports .png, .keras, .csv, etc.).
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    return os.path.join(SAVE_DIR, filename)

def get_timestamped_filename(filename):
    """
    Return a timestamped filename, useful for versioning.
    """
    base, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{base}_{timestamp}{ext}"

def save_json(obj, filename):
    """
    Save an object as a JSON file to the specified path.
    """
    with open(get_save_path(filename), 'w') as f:
        json.dump(obj, f, indent=4)

def save_pickle(obj, filename):
    """
    Save an object as a Pickle file to the specified path.
    """
    with open(get_save_path(filename), 'wb') as f:
        pickle.dump(obj, f)
        
# %%

# %%
