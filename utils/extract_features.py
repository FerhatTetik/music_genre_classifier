import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=30)

    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rmse = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs_mean = np.mean(mfccs, axis=1)

    return np.hstack([
        chroma_stft, rmse, spectral_centroid,
        spectral_bandwidth, rolloff, zero_crossing_rate,
        mfccs_mean
    ]) 