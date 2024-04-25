import os
import librosa
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Directory where your .wav files are located
wav_dir = r'C:\BatSignal\BatData'

# Define a directory to save the spectrograms
spectrogram_dir = r'C:\BatSignal\Spectrogram'

# List to hold spectrograms and labels
spectrograms = []
labels = []

# Define a fixed size for your spectrograms
fixed_size = (640, 640)

# Loop over all directories in the parent directory
for species_dir in ['Parsed_Capuchinbird_Clips', 'Parsed_Not_Capuchinbird_Clips']:
    # Define the subdirectory where .wav files are located
    wav_subdir = os.path.join('NonBatCall', species_dir)

    # Create a directory for the spectrogram images of this species
    # spectrogram_dir_species = os.path.join(spectrogram_dir, species_dir)
    # os.makedirs(spectrogram_dir_species, exist_ok=True)

    # Loop over all files in the subdirectory
    for filename in os.listdir(wav_subdir):
        if filename.endswith('.wav'):
            # Load .wav file
            y, sr = librosa.load(os.path.join(wav_subdir, filename))

            # Convert to spectrogram
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

            # Resize spectrogram to fixed size using interpolation
            zoom_factor = (fixed_size[0] / spectrogram.shape[0], fixed_size[1] / spectrogram.shape[1])
            spectrogram = scipy.ndimage.zoom(spectrogram, zoom_factor)

            # Flatten spectrogram and add to list
            spectrograms.append(spectrogram.flatten())

            # Add species to labels list
            labels.append(species_dir)

            # Save spectrogram as an image
            # plt.imshow(spectrogram, cmap='inferno')
            # plt.axis('off')
            # plt.savefig(os.path.join(spectrogram_dir_species, f'{filename}.png'))

            print(f'{filename} converted to spectrogram.')

# Convert list to numpy array
spectrograms = np.array(spectrograms)

# Normalize data
spectrograms = (spectrograms - np.min(spectrograms)) / (np.max(spectrograms) - np.min(spectrograms))

# Convert labels to integers
label_dict = {label: i for i, label in enumerate(set(labels))}
labels = [label_dict[label] for label in labels]

# Convert labels to categorical
labels = to_categorical(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

# Reshape data to include channel dimension
X_train = X_train.reshape(-1, fixed_size[0], fixed_size[1], 1)
X_test = X_test.reshape(-1, fixed_size[0], fixed_size[1], 1)

# Initialize the model
model = Sequential()

# Add a convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(fixed_size[0], fixed_size[1], 1)))

# Add a pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the tensor output from the previous layer
model.add(Flatten())

# Add a dense layer
model.add(Dense(128, activation='relu'))

# Add the output layer
model.add(Dense(len(label_dict), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model
model.save(r'C:\BatSignal\OutputModelBirdCall10Epoch.h5')