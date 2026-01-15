import torch
import numpy as np
from pathlib import Path
from src.utils.npy_loader import NpyFile

# Raw load
raw = np.load('data/darcy_continuous/smh_train/coeff.npy')
print(f"Raw shape: {raw.shape}")

# Method 1: Your debug script
a1 = raw.reshape(-1, 1000).T  # (1000, 841)
print(f"Method 1 reshape: {a1.shape}")
print(f"Method 1 sample 0, first 5 values: {a1[0, :5]}")

# Method 2: _load_data style  
data = NpyFile(path=Path('data/darcy_continuous/smh_train/'), mode='r')
a2 = np.array(data["coeff"]).T  # What shape?
print(f"Method 2 after .T: {a2.shape}")
print(f"Method 2 sample 0, first 5 values: {a2[0, :5]}")
