from flask import Flask, request, render_template
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os

UPLOAD_FOLDER = 'static/file/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('Sound.h5')

# Define classes for the predictions
classes = ['Autisim mild', 'Autisim moderate', 'normal']  # Replace with your actual class names

def predict_class(audio_file):
    # Load and process the audio file
    y, sr = librosa.load(audio_file, sr=None)
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    # Pad or truncate the features to match the expected shape
    features_processed = np.pad(features, ((0, 0), (0, 2508 - features.shape[1])), mode='constant')[:, :, np.newaxis]

    # Make predictions
    predictions = model.predict(np.expand_dims(features_processed, axis=0))
    predicted_class = classes[np.argmax(predictions)]

    return predicted_class

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index1.html', prediction="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index1.html', prediction="No selected file")

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    predicted_class = predict_class(file_path)

    return render_template('index1.html', prediction=predicted_class, wavfile=file.filename)


if __name__ == '__main__':
    app.run(debug=True)
