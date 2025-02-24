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
notes_df = pd.read_csv(csv_path)

# Check the columns to ensure 'Instrument (in full)' exists
print("Columns in DataFrame:", notes_df.columns)

# Remove extra spaces from column names (if any)
notes_df.columns = notes_df.columns.str.strip()

# Use 'Instrument (in full)' for the labels
if 'Instrument (in full)' not in notes_df.columns:
    raise ValueError("The 'Instrument (in full)' column does not exist in the CSV file. Please check the column names.")

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
notes_df['Spectrogram'] = notes_df['Path'].apply(lambda x: generate_spectrogram(x))

# Remove rows with missing spectrograms
notes_df = notes_df.dropna(subset=['Spectrogram'])

# Convert the 'Spectrogram' column from list to numpy ndarray
notes_df['Spectrogram'] = notes_df['Spectrogram'].apply(lambda x: np.array(x))

# Function to preprocess spectrograms: padding or cropping them to the target shape
def preprocess_spectrogram(spectrogram, target_shape=(256, 256)):
    # Crop or pad the spectrogram to have a consistent size of (256, 256)
    if spectrogram.shape[1] > target_shape[1]:
        # Crop the spectrogram if it's larger than the target width
        spectrogram = spectrogram[:, :target_shape[1]]
    elif spectrogram.shape[1] < target_shape[1]:
        # Pad the spectrogram if it's smaller than the target width
        pad_width = target_shape[1] - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    if spectrogram.shape[0] > target_shape[0]:
        # Crop the spectrogram if it's larger than the target height
        spectrogram = spectrogram[:target_shape[0], :]
    elif spectrogram.shape[0] < target_shape[0]:
        # Pad the spectrogram if it's smaller than the target height
        pad_height = target_shape[0] - spectrogram.shape[0]
        spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)), mode='constant', constant_values=0)
    return spectrogram

# Apply the preprocessing function to all spectrograms in the DataFrame
notes_df['Processed_Spectrogram'] = notes_df['Spectrogram'].apply(preprocess_spectrogram)

# Convert the processed spectrograms into a numpy array
X_spect = np.array([i for i in notes_df['Processed_Spectrogram']])

# Reshape to add the channel dimension (1 for grayscale)
X_spect = X_spect.reshape(X_spect.shape + (1,))

# Encode the labels ('Instrument (in full)' column)
y = notes_df['Instrument (in full)'].values

# Use LabelEncoder to convert labels to numeric values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Convert labels to one-hot encoding
y_onehot = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_spect, y_onehot, test_size=0.2, random_state=42)

# Model architecture (Convolutional Neural Network)
pitch_model = Sequential()

# First Convolutional Block
pitch_model.add(Conv2D(24, (3, 3), padding='same', input_shape=X_train.shape[1:]))
pitch_model.add(LeakyReLU())
pitch_model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Block
pitch_model.add(Conv2D(48, (3, 3), padding='same'))
pitch_model.add(LeakyReLU())
pitch_model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Block
pitch_model.add(Conv2D(96, (3, 3), padding='same'))
pitch_model.add(LeakyReLU())
pitch_model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the output of convolutional layers
pitch_model.add(Flatten())

# Fully connected layer
pitch_model.add(Dense(128))
pitch_model.add(LeakyReLU())

# Output layer with softmax activation for multi-class classification
pitch_model.add(Dense(y_onehot.shape[1], activation='softmax'))

# Compile the model
pitch_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = pitch_model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# Evaluate the model on the test set
test_loss, test_acc = pitch_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save the trained model
pitch_model.save('pitch_detection_model.h5')

print("Model saved to 'pitch_detection_model.h5'")
