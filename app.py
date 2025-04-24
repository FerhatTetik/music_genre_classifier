from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import joblib
import tensorflow as tf
from utils.extract_features import extract_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# uploads klasörü yoksa oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model ve yardımcı nesneleri yükle
model = tf.keras.models.load_model("models/model_20250423_2155.h5")
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler_193.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    genre = None
    confidence = None
    if request.method == "POST":
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                features = extract_features(filepath)
                features_scaled = scaler.transform(features.reshape(1, -1))

                predictions = model.predict(features_scaled)
                predicted_index = predictions.argmax()

                genre = label_encoder.inverse_transform([predicted_index])[0]
                confidence = float(predictions[0][predicted_index])

                os.remove(filepath)

            except Exception as e:
                print(f"Hata: {str(e)}")
                genre = "Hata oluştu"
                confidence = 0.0

    return render_template("index.html", genre=genre, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
