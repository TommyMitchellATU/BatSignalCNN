import os
from flask import Flask, render_template, request
from keras.models import load_model
import librosa
import numpy as np
import scipy.ndimage

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = load_model(r'C:\BatSignal\batsignal\CNNUsingTensorflow\Model\my_model.h5')

# Define a fixed size for your spectrograms
fixed_size = (640, 640)

# Define your label dictionary
label_dict = {0: 'NYCLEI', 1: 'PIPPIP'}

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.wav'):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load .wav file
        y, sr = librosa.load(file_path)

        # Convert to spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        # Resize spectrogram to fixed size using interpolation
        zoom_factor = (fixed_size[0] / spectrogram.shape[0], fixed_size[1] / spectrogram.shape[1])
        spectrogram = scipy.ndimage.zoom(spectrogram, zoom_factor)

        # Flatten spectrogram
        spectrogram = spectrogram.flatten()

        # Normalize spectrogram
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

        # Reshape spectrogram to include channel dimension
        spectrogram = spectrogram.reshape(1, fixed_size[0], fixed_size[1], 1)

        # Make prediction
        y_pred = model.predict(spectrogram)

        # Get confidence percentages
        confidence_NYCLEI = y_pred[0][0] * 100
        if len(y_pred[0]) > 1:
            confidence_PIPPIP = y_pred[0][1] * 100
        else:
            confidence_PIPPIP = 0

        return f' Confidence - NYCLEI: {confidence_NYCLEI}%, PIPPIP: {confidence_PIPPIP}%'
    else:
        return 'Invalid file format'

if __name__ == "__main__":
    app.run(debug=True)