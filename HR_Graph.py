import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_rows_num = 2126

dataset_path = 'datasets/fetal_health.csv'
data = pd.read_csv(dataset_path)

# Time Setup
time_duration = 10 * 60  # 10 minutes in seconds
sampling_rate = 1  # 1 Hz sampling (1 sample per second)
time = np.arange(0, time_duration, sampling_rate)  # Time vector

#CTG Parameters
baseline_vector = data['baseline value'].values #BPM

STV_vector = data['mean_value_of_short_term_variability'].values #BPM
LTV_vector = data['mean_value_of_long_term_variability'].values  #BPM
LTV_frequency = 1 / 60  # Long-term variability frequency (1 cycle per minute)

acceleration_vector = data['accelerations'].values
deceleration_vector = data['light_decelerations'].values

fetal_movement_vector = data['fetal_movement'].values
uterine_contractions_vector = data['uterine_contractions'].values

fhr_vector_of_vectors = np.zeros(dataset_rows_num, dtype=np.ndarray)

for n in range(dataset_rows_num):
    fhr_vector_of_vectors[n] = baseline_vector[n] + np.random.uniform(-STV_vector[n], STV_vector[n], size=len(time))
    
    fhr_vector_of_vectors[n] += LTV_vector[n] * np.sin(2*np.pi*LTV_frequency*time)
    
    fhr_vector_of_vectors[n] += acceleration_vector[n]
    fhr_vector_of_vectors[n] -= deceleration_vector[n]
    
    fhr_vector_of_vectors[n] +=fetal_movement_vector[n] * np.random.uniform(-5, 5, size=len(time))
    
    fhr_vector_of_vectors[n] += uterine_contractions_vector[n] * np.sin(2*np.pi*(1/300)*time)
    


# Plot the reconstructed fhr_vector_of_vectors
plt.figure(figsize=(12, 6))
plt.plot(time / 60, fhr_vector_of_vectors[1000], label="Instantaneous fhr_vector_of_vectors", color="blue")
plt.axhline(baseline_vector[1000], color="red", linestyle="--", label="Baseline fhr_vector_of_vectors")
plt.title("Reconstructed Instantaneous Fetal Heart Rate (fhr_vector_of_vectors)")
plt.xlabel("Time (minutes)")
plt.ylabel("Fetal Heart Rate (bpm)")
plt.legend()
plt.grid()
plt.show()
