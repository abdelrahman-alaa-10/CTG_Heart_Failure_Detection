import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QVBoxLayout, QLabel
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scipy.signal as signal

class main(QMainWindow):
    def __init__(self):
        super(main, self).__init__()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        loadUi("main.ui", self)

        self.original_graph = self.findChild(QFrame, "original_graph")
        self.filtered_graph = self.findChild(QFrame, "filtered_graph")

        self.mean_label = self.findChild(QLabel, "mean")
        self.median_label = self.findChild(QLabel, "median")
        self.max_label = self.findChild(QLabel, "max")
        self.min_label = self.findChild(QLabel, "min")
        self.hrv_label = self.findChild(QLabel, "hrv_label")
        self.fhr_label = self.findChild(QLabel, "fhr_label")
        self.status_label = self.findChild(QLabel, "status")

        self.draw_original_graph()
        self.draw_filtered_graph()

    def draw_original_graph(self):
        dataset_path = 'fetal_health.csv'
        data = pd.read_csv(dataset_path)

        dataset_rows_num = 2126
        time_duration = 10 * 60  # 10 minutes in seconds
        sampling_rate = 1  # 1 Hz sampling (1 sample per second)
        time = np.arange(0, time_duration, sampling_rate)  # Time vector

        baseline_vector = data['baseline value'].values  # BPM
        STV_vector = data['mean_value_of_short_term_variability'].values  # BPM
        LTV_vector = data['mean_value_of_long_term_variability'].values  # BPM
        LTV_frequency = 1 / 60  # Long-term variability frequency (1 cycle per minute)

        acceleration_vector = data['accelerations'].values
        deceleration_vector = data['light_decelerations'].values
        fetal_movement_vector = data['fetal_movement'].values
        uterine_contractions_vector = data['uterine_contractions'].values

        fhr_vector_of_vectors = np.zeros(dataset_rows_num, dtype=np.ndarray)

        for n in range(dataset_rows_num):
            fhr_vector_of_vectors[n] = baseline_vector[n] + np.random.uniform(-STV_vector[n], STV_vector[n], size=len(time))
            fhr_vector_of_vectors[n] += LTV_vector[n] * np.sin(2 * np.pi * LTV_frequency * time)
            fhr_vector_of_vectors[n] += acceleration_vector[n]
            fhr_vector_of_vectors[n] -= deceleration_vector[n]
            fhr_vector_of_vectors[n] += fetal_movement_vector[n] * np.random.uniform(-5, 5, size=len(time))
            fhr_vector_of_vectors[n] += uterine_contractions_vector[n] * np.sin(2 * np.pi * (1 / 300) * time)

        selected_signal_index = 1000  # Choose the 1000th sample for demonstration
        original_signal = fhr_vector_of_vectors[selected_signal_index]

        # Calculate statistics
        mean_value = np.mean(original_signal)
        median_value = np.median(original_signal)
        max_value = np.max(original_signal)
        min_value = np.min(original_signal)

        self.hrv_label.setText("")

        # Set label texts
        self.mean_label.setText(f"{mean_value:.2f}")
        self.median_label.setText(f"{median_value:.2f}")
        self.max_label.setText(f"{max_value:.2f}")
        self.min_label.setText(f"{min_value:.2f}")

        fig, ax = plt.subplots()
        ax.plot(time / 60, original_signal, label="Original Signal", color="blue", alpha=0.7)
        ax.axhline(baseline_vector[selected_signal_index], color="red", linestyle="--", label="Baseline fhr_vector_of_vectors")
        ax.set_title("Original Fetal Heart Rate Signal")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Fetal Heart Rate (bpm)")
        fig.patch.set_facecolor('#fdf0d5')  # Set the figure background color
        ax.set_facecolor('#fdf0d5')  # Set the axes background color

        ax.legend()
        ax.grid()

        canvas = FigureCanvas(fig)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        self.original_graph.setLayout(layout)

    def draw_filtered_graph(self):
        dataset_path = 'fetal_health.csv'
        data = pd.read_csv(dataset_path)

        dataset_rows_num = 2126
        time_duration = 10 * 60  # 10 minutes in seconds
        sampling_rate = 1  # 1 Hz sampling (1 sample per second)
        time = np.arange(0, time_duration, sampling_rate)  # Time vector

        baseline_vector = data['baseline value'].values  # BPM
        STV_vector = data['mean_value_of_short_term_variability'].values  # BPM
        LTV_vector = data['mean_value_of_long_term_variability'].values  # BPM
        LTV_frequency = 1 / 60  # Long-term variability frequency (1 cycle per minute)

        acceleration_vector = data['accelerations'].values
        deceleration_vector = data['light_decelerations'].values
        fetal_movement_vector = data['fetal_movement'].values
        uterine_contractions_vector = data['uterine_contractions'].values

        fhr_vector_of_vectors = np.zeros(dataset_rows_num, dtype=np.ndarray)

        for n in range(dataset_rows_num):
            fhr_vector_of_vectors[n] = baseline_vector[n] + np.random.uniform(-STV_vector[n], STV_vector[n], size=len(time))
            fhr_vector_of_vectors[n] += LTV_vector[n] * np.sin(2 * np.pi * LTV_frequency * time)
            fhr_vector_of_vectors[n] += acceleration_vector[n]
            fhr_vector_of_vectors[n] -= deceleration_vector[n]
            fhr_vector_of_vectors[n] += fetal_movement_vector[n] * np.random.uniform(-5, 5, size=len(time))
            fhr_vector_of_vectors[n] += uterine_contractions_vector[n] * np.sin(2 * np.pi * (1 / 300) * time)

        selected_signal_index = 1000  # Choose the 1000th sample for demonstration
        original_signal = fhr_vector_of_vectors[selected_signal_index]

        def butter_lowpass_filter(data, cutoff, fs, order=4):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            filtered_data = signal.filtfilt(b, a, data)
            return filtered_data

        cutoff_frequency = 0.2  # Cutoff frequency in Hz
        sampling_frequency = sampling_rate  # Sampling frequency in Hz

        filtered_signal = butter_lowpass_filter(original_signal, cutoff_frequency, sampling_frequency)

        fig, ax = plt.subplots()
        ax.plot(time / 60, filtered_signal, label="Filtered Signal", color="green", linestyle="--", linewidth=2)
        ax.axhline(baseline_vector[selected_signal_index], color="red", linestyle="--", label="Baseline fhr_vector_of_vectors")
        ax.set_title("Filtered Fetal Heart Rate Signal")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Fetal Heart Rate (bpm)")
        fig.patch.set_facecolor('#fdf0d5')  # Set the figure background color
        ax.set_facecolor('#fdf0d5')  # Set the axes background color
        ax.legend()
        ax.grid()

        canvas = FigureCanvas(fig)
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        self.filtered_graph.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = main()
    window.show()
    window.showMaximized()
    sys.exit(app.exec_())