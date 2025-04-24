import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from utils.extract_features import extract_features

# Dataset dizini
DATASET_DIR = "dataset"

# Özellikler ve etiketler listesi
features_list = []
genres = []

# Her tür klasörünü gez
for genre in os.listdir(DATASET_DIR):
    genre_path = os.path.join(DATASET_DIR, genre)
    if os.path.isdir(genre_path):
        for filename in os.listdir(genre_path):
            if filename.endswith(".mp3"):
                file_path = os.path.join(genre_path, filename)
                features = extract_features(file_path)
                features_list.append(features)
                genres.append(genre)

# Özellikleri NumPy array'e çevir
X = np.array(features_list)

# StandardScaler eğitimi
scaler = StandardScaler()
scaler.fit(X)

# Scaler'ı kaydet
joblib.dump(scaler, "scaler_193.pkl")
print("Scaler başarıyla eğitildi ve kaydedildi.")
