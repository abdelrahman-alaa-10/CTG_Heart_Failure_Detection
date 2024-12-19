import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal

# Load Dataset
dataset_path = 'fetal_health.csv'
data = pd.read_csv(dataset_path)

# Parameters
dataset_rows_num = 2126
time_duration = 10 * 60  # 10 minutes in seconds
sampling_rate = 1  # 1 Hz sampling (1 sample per second)
time = np.arange(0, time_duration, sampling_rate)  # Time vector

# CTG Parameters
baseline_vector = data['baseline value'].values  # BPM
STV_vector = data['mean_value_of_short_term_variability'].values  # BPM
LTV_vector = data['mean_value_of_long_term_variability'].values  # BPM
LTV_frequency = 1 / 60  # Long-term variability frequency (1 cycle per minute)

acceleration_vector = data['accelerations'].values
deceleration_vector = data['light_decelerations'].values
fetal_movement_vector = data['fetal_movement'].values
uterine_contractions_vector = data['uterine_contractions'].values

# Generate fhr_vector_of_vectors
fhr_vector_of_vectors = np.zeros(dataset_rows_num, dtype=np.ndarray)

for n in range(dataset_rows_num):
    fhr_vector_of_vectors[n] = baseline_vector[n] + np.random.uniform(-STV_vector[n], STV_vector[n], size=len(time))
    fhr_vector_of_vectors[n] += LTV_vector[n] * np.sin(2 * np.pi * LTV_frequency * time)
    fhr_vector_of_vectors[n] += acceleration_vector[n]
    fhr_vector_of_vectors[n] -= deceleration_vector[n]
    fhr_vector_of_vectors[n] += fetal_movement_vector[n] * np.random.uniform(-5, 5, size=len(time))
    fhr_vector_of_vectors[n] += uterine_contractions_vector[n] * np.sin(2 * np.pi * (1 / 300) * time)


case_num = 5
# Define a Butterworth Low-Pass Filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

# Filtering Parameters
cutoff_frequency = 0.2  # Cutoff frequency in Hz
sampling_frequency = sampling_rate  # Sampling frequency in Hz

# Select a Signal to Filter
selected_signal_index = 1000  # Choose the 1000th sample for demonstration
original_signal = fhr_vector_of_vectors[selected_signal_index]
filtered_signal = butter_lowpass_filter(original_signal, cutoff_frequency, sampling_frequency)

# Plot Original and Filtered Signals
plt.figure(figsize=(12, 6))
plt.plot(time / 60, fhr_vector_of_vectors[case_num], label="Instantaneous fhr_vector_of_vectors", color="blue")
plt.axhline(baseline_vector[case_num], color="red", linestyle="--", label="Baseline fhr_vector_of_vectors")
plt.title("Reconstructed Instantaneous Fetal Heart Rate (fhr_vector_of_vectors)")

plt.plot(time / 60, original_signal, label="Original Signal", color="blue", alpha=0.7)
plt.plot(time / 60, filtered_signal, label="Filtered Signal", color="green", linestyle="--", linewidth=2)
plt.axhline(baseline_vector[selected_signal_index], color="red", linestyle="--", label="Baseline fhr_vector_of_vectors")
plt.title("Fetal Heart Rate Signal Before and After Filtering")
plt.xlabel("Time (minutes)")
plt.ylabel("Fetal Heart Rate (bpm)")
plt.legend()
plt.grid()
plt.show()
