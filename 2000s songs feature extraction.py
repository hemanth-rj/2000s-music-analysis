import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
from scipy.signal import find_peaks

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    chroma_index = chroma_mean.argmax()
    chroma_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = chroma_notes[chroma_index]
    return key

def compute_rmse(y):
    rmse = librosa.feature.rms(y=y)
    return np.mean(rmse)

def dynamic_range(y, sr):

    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    rms_db = rms_db[np.isfinite(rms_db)]

    if len(rms_db) == 0:
        return 0.0

    dynamic_range_db = np.percentile(rms_db, 90) - np.percentile(rms_db, 10)

    return dynamic_range_db

def compute_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)

def compute_spectral_bandwidth(y, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.mean(bandwidth)

def compute_tempo(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return tempo.item()

folder_path = r'D:\xyz\Abroad Univ\Project\APT\Songs'

results = []

for filename in os.listdir(folder_path):
    if filename.endswith(".mp3") or filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")

        y, sr = librosa.load(file_path)

        key = detect_key(y, sr)
        rmse = compute_rmse(y)
        dynamic_range_value = dynamic_range(y, sr)
        spectral_centroid = compute_spectral_centroid(y, sr)
        spectral_bandwidth = compute_spectral_bandwidth(y, sr)
        tempo = compute_tempo(y, sr)

        results.append({
            'Song': filename,
            'Key': key,
            'RMSE': rmse,
            'Dynamic Range': dynamic_range_value,
            'Spectral Centroid': spectral_centroid,
            'Spectral Bandwidth': spectral_bandwidth,
            'Tempo': tempo
        })

df = pd.DataFrame(results)

csv_file_path = 'song_analysis_results.csv'
df.to_csv(csv_file_path, index=False)

print(f"Analysis complete! Results saved to {csv_file_path}")
