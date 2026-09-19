from pathlib import Path
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("ecg-arrhythmia-data")

mat_files = sorted(ROOT.rglob("*.mat"))
mat_path = mat_files[0]

print("Loading:", mat_path)
data = loadmat(mat_path)
ecg = data["val"]

print("ECG shape:", ecg.shape)

sampling_rate = 500

lead = ecg[0]

time = np.arange(len(lead)) / sampling_rate


#get some baseline
print("Minimum:", np.min(lead))
print("Maximum:", np.max(lead))
print("Mean:", np.mean(lead))
print("Standard deviation:", np.std(lead))


#mean and std-deviation for normalization [finally used this in project]

mean = np.mean(lead)
std_dev = np.std(lead)

normalized_lead = (lead - mean) / std_dev

print("normalized mean: ", np.mean(normalized_lead))
print("normalized standard deviation: ", np.std(normalized_lead))


#graphing normallized and og signals

# Compare original and normalized signals
plt.figure(figsize=(14, 8))

# Original signal
plt.subplot(2, 1, 1)
plt.plot(time, lead)
plt.title("Original ECG - Lead I")
plt.xlabel("Time (seconds)")
plt.ylabel("Raw Amplitude")
plt.grid(True)

# Normalized signal
plt.subplot(2, 1, 2)
plt.plot(time, normalized_lead)
plt.title("Normalized ECG - Lead I")
plt.xlabel("Time (seconds)")
plt.ylabel("Standardized Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()