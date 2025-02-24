import pandas as pd
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU

# Load the metadata CSV containing file paths and instrument names
csv_path = 'C:/Users/hp/Downloads/New_folder/TinySOL_metadata.csv'
instruments_df = pd.read_csv(csv_path)

# Define the directory where your audio files are located
audio_dir = 'C:/Users/hp/Downloads/New_folder/TinySOL'

# Function to generate spectrogram from audio file
def generate_spectrogram(file_path):
    # Check if the file exists
    full_path = os.path.join(audio_dir, file_path)
    if os.path.exists(full_path):
        # Load the audio file
        y, sr = librosa.load(full_path, sr=None)
        # Generate the spectrogram (STFT)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        return D
    else:
        return None  # If file doesn't exist, return None

# Apply the function to each file path in the DataFrame and store the result in a new 'Spectrogram' column
instruments_df['Spectrogram'] = instruments_df['Path'].apply(lambda x: generate_spectrogram(x))

# Remove rows with missing spectrograms
instruments_df = instruments_df.dropna(subset=['Spectrogram'])

# Convert the 'Spectrogram' column from list to numpy ndarray
instruments_df['Spectrogram'] = instruments_df['Spectrogram'].apply(lambda x: np.array(x))

# Function to preprocess spectrograms: padding or cropping them to the target shape
def preprocess_spectrogram(spectrogram, target_shape=(1025, 248)):
    # Crop or pad the spectrogram to have a consistent size of (1025, 248)
    if spectrogram.shape[1] > target_shape[1]:
        # Crop the spectrogram if it's larger than the target width
        spectrogram = spectrogram[:, :target_shape[1]]
    elif spectrogram.shape[1] < target_shape[1]:
        # Pad the spectrogram if it's smaller than the target width
        pad_width = target_shape[1] - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    return spectrogram

# Apply the preprocessing function to all spectrograms in the DataFrame
instruments_df['Processed_Spectrogram'] = instruments_df['Spectrogram'].apply(preprocess_spectrogram)

# Convert the processed spectrograms into a numpy array
X_spect = np.array([i for i in instruments_df['Processed_Spectrogram']])

# Reshape to add the channel dimension (1 for grayscale)
X_spect = X_spect.reshape(X_spect.shape + (1,))

# Verify the shape of the final dataset
print(f"Shape of X_spect: {X_spect.shape}")  # Should be (number_of_samples, 1025, 248, 1)

# Initialize the LabelEncoder
le = LabelEncoder()

# Fit and transform the instrument names into numeric labels
y_inst = le.fit_transform(instruments_df['Instrument (in full)'])

# Convert the labels to one-hot encoding
y = to_categorical(y_inst, num_classes=14)  # 14 is the number of classes (instruments)

# Split into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_spect, y, test_size=0.2, random_state=42)

# Print the shapes of the datasets to verify
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Define the model
spectrogram_shape = (1025, 248, 1)  # Spectrogram shape after reshaping

model = Sequential()

# Add first Conv2D layer
model.add(Conv2D(23, kernel_size=(3, 3), activation='linear', input_shape=spectrogram_shape, padding='same'))
model.add(LeakyReLU(alpha=0.1))

# Add MaxPooling2D
model.add(MaxPooling2D((2, 2), padding='same'))

# Add second Conv2D layer
model.add(Conv2D(43, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))

# Add MaxPooling2D
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Add third Conv2D layer
model.add(Conv2D(86, (3, 3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))

# Add MaxPooling2D
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Flatten the output before passing it to fully connected layers
model.add(Flatten())

# Add fully connected layer with 86 neurons
model.add(Dense(86, activation='linear'))
model.add(LeakyReLU(alpha=0.1))

# Add output layer with 14 classes for instrument categories (softmax for multi-class classification)
model.add(Dense(14, activation='softmax'))

# Summarize the model
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the trained model to a file (e.g., model.h5)
model.save('your_trained_model.h5')  # Replace 'your_trained_model.h5' with your desired filename
print("Model saved successfully!")

import joblib

# Save the LabelEncoder to a file (e.g., label_encoder.pkl)
joblib.dump(le, 'label_encoder.pkl')  # Replace 'label_encoder.pkl' with your desired filename
print("LabelEncoder saved successfully!")