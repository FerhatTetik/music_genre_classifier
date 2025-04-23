from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import joblib
import tensorflow as tf
from utils.extract_features import extract_features

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Modeli ve etiket kodlayıcıyı yükle
model = tf.keras.models.load_model("model/genre_model.h5")
label_encoder = joblib.load("model/label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    genre = None
    if request.method == "POST":
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Özellikleri çıkar ve yeniden şekillendir
            features = extract_features(filepath).reshape(1, -1)
            
            # Tahmin yap
            predictions = model.predict(features)
            predicted_index = predictions.argmax()
            genre = label_encoder.inverse_transform([predicted_index])[0]

    return render_template("index.html", genre=genre)

if __name__ == "__main__":
    app.run(debug=True) 