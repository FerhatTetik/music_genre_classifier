import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.extract_features import extract_features

DATASET_PATH = 'dataset/'  # klasör içinde her klasör bir tür: pop, arabesk, klasik

def load_data():
    features = []
    labels = []

    for genre in os.listdir(DATASET_PATH):
        genre_folder = os.path.join(DATASET_PATH, genre)
        if not os.path.isdir(genre_folder):
            continue

        for filename in os.listdir(genre_folder):
            if filename.endswith('.wav') or filename.endswith('.mp3'):
                file_path = os.path.join(genre_folder, filename)
                try:
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(genre)
                except Exception as e:
                    print(f"Hata: {file_path} - {e}")

    return np.array(features), np.array(labels)

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_save():
    X, y = load_data()
    
    # Etiketleri sayısallaştır
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # One-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    
    # Veriyi böl
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
    
    # Modeli oluştur
    model = create_model(X.shape[1], len(label_encoder.classes_))
    
    # Modeli derle
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Early stopping ekle
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Modeli eğit
    history = model.fit(X_train, y_train,
                       epochs=30,
                       batch_size=16,
                       validation_data=(X_test, y_test),
                       callbacks=[early_stopping])
    
    # Test seti üzerinde değerlendir
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest doğruluğu: {test_acc:.2f}")
    
    # Modeli ve etiket kodlayıcıyı kaydet
    model.save("model/genre_model.h5")
    joblib.dump(label_encoder, "model/label_encoder.pkl")
    print("Model kaydedildi!")

if __name__ == "__main__":
    train_and_save() 