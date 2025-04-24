import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder
from utils.extract_features import extract_features

# Dataset yolu
dataset_path = "dataset"

# Veri ve etiketleri topla
X = []
y = []

genres = os.listdir(dataset_path)
for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            if file.endswith(".mp3"):
                file_path = os.path.join(genre_path, file)
                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(genre)
                except Exception as e:
                    print(f"Hata: {file_path} - {str(e)}")

X = np.array(X)
y = np.array(y)

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "models/label_encoder.pkl")

# Ölçekleme
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "models/scaler.pkl")

# Model tanımı
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y_encoded, epochs=30, batch_size=16)

# Modeli kaydet
model.save("models/model_20250423_2155.h5")
print("Model yeniden eğitildi ve kaydedildi.")
