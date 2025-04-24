import librosa
import numpy as np

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, mono=True, duration=30)
        
        if len(y) == 0:
            raise ValueError("Ses dosyası boş")

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        harmony = librosa.effects.harmonic(y)
        perceptr = librosa.feature.spectral_contrast(y=harmony, sr=sr)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        features = []

        features.extend([
            float(np.mean(chroma_stft)), float(np.std(chroma_stft)),
            float(np.mean(rmse)), float(np.std(rmse)),
            float(np.mean(spec_cent)), float(np.std(spec_cent)),
            float(np.mean(spec_bw)), float(np.std(spec_bw)),
            float(np.mean(rolloff)), float(np.std(rolloff)),
            float(np.mean(zcr)), float(np.std(zcr)),
            float(tempo)
        ])

        for e in mfcc:
            features.extend([
                float(np.mean(e)), float(np.std(e)),
                float(np.min(e)), float(np.max(e))
            ])

        for e in perceptr:
            features.extend([float(np.mean(e)), float(np.std(e))])

        features.extend([
            float(np.mean(mel_spec_db)),
            float(np.std(mel_spec_db)),
            float(np.min(mel_spec_db)),
            float(np.max(mel_spec_db))
        ])

        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        features = np.array(features)
        if len(features) != 193:
            features = np.pad(features, (0, 193 - len(features)), 'constant')

        return features

    except Exception as e:
        print(f"Hata: {file_path} - {str(e)}")
        return np.zeros(193)
