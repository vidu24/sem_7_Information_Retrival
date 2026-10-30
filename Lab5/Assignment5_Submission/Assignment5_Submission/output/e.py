from pathlib import Path

import numpy as np

# Load the array from the file in the same folder as this script
array_path = Path(__file__).with_name("embeddings.npy")
data = np.load(array_path)

# Inspect structure and sample values
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("\nFirst vector (first 10 elements):")
print(data[0, :10])