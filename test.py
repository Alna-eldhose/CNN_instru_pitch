import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import librosa.display  # For plotting the spectrogram

# Function to generate the spectrogram from the wav file
def generate_spectrogram(file_path, target_shape):
    y, sr = librosa.load(file_path, sr=None)  # Load the audio file
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # Generate the spectrogram

    # Resize or pad the spectrogram to match the model's expected input shape
    D_resized = np.zeros(target_shape)  # Create a blank array of the target shape
    h, w = min(D.shape[0], target_shape[0]), min(D.shape[1], target_shape[1])
    D_resized[:h, :w] = D[:h, :w]  # Copy over data into the correct dimensions

    # Add channel dimension for compatibility with Conv2D layers
    D_resized = D_resized.reshape(target_shape + (1,))
    return D_resized

# Load the trained model for instrument classification
pitch_model = load_model('C:/Users/hp/Downloads/New_folder/pitch_detection_model.h5')  # Replace with your actual pitch detection model path
instrument_model = load_model('C:/Users/hp/Downloads/New_folder/your_trained_model.h5')  # Replace with your model path

# Print the model's input shape for debugging
print("Instrument model input shape:", instrument_model.input_shape)
print("Pitch model input shape:", pitch_model.input_shape)

# Load the LabelEncoder (if used)
le = joblib.load('label_encoder.pkl')  # Replace with your LabelEncoder path

# Function to detect pitch (i.e., frequency)
def detect_pitch(file_path):
    # Get the input shape of the pitch model
    target_shape = pitch_model.input_shape[1:3]  # Extract (height, width)

    # Generate the spectrogram for the input file
    spectrogram = generate_spectrogram(file_path, target_shape)

    # Reshape the spectrogram to match the expected input shape for the pitch model
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension

    # If the pitch model expects a specific input shape, such as (256, 256, 1), reshape the spectrogram
    if spectrogram.shape[1] != 256 or spectrogram.shape[2] != 256:
        spectrogram = np.resize(spectrogram, (1, 256, 256, 1))  # Resize if necessary

    # Predict pitch using the pitch detection model
    pitch_prediction = pitch_model.predict(spectrogram)
    
    # Assuming pitch_prediction is the predicted frequency in Hz
    pitch_frequency = pitch_prediction[0][0]  # Get the predicted frequency value (first prediction)
    return pitch_frequency

# Function to test a single wav file
def test_wav_file(file_path):
    # Get the input shape of the instrument model
    target_shape = instrument_model.input_shape[1:3]  # Extract (height, width)

    # Generate the spectrogram for the input file
    spectrogram = generate_spectrogram(file_path, target_shape)

    # Reshape it to the expected input shape for the instrument model
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Add batch dimension

    # Predict using the instrument classification model
    instrument_prediction = instrument_model.predict(spectrogram)

    # Get the predicted class index for the instrument
    predicted_class_index = np.argmax(instrument_prediction)

    # Map the index back to the instrument name
    predicted_label = le.inverse_transform([predicted_class_index])[0]

    # Output the instrument prediction
    print(f"Predicted instrument: {predicted_label}")

    # Detect pitch (frequency) for the same file
    pitch_frequency = detect_pitch(file_path)

    # Convert the pitch frequency to musical pitch (e.g., A4, C5, etc.)
    pitch_name = frequency_to_pitch(pitch_frequency)

    # Output the detected pitch
    print(f"Detected pitch: {pitch_name} ({pitch_frequency} Hz)")

    # Optionally: Display the spectrogram for the input file
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(librosa.load(file_path)[0])), ref=np.max),
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram of {file_path}")
    plt.show()

# Function to convert frequency (in Hz) to musical pitch (e.g., A1, A2, etc.)
def frequency_to_pitch(frequency):
    if frequency <= 0:
        return None  # Invalid frequency
    
    # MIDI number calculation based on frequency
    midi_number = 69 + 12 * np.log2(frequency / 440)
    
    # Calculate the note name
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_index = int(round(midi_number)) % 12
    octave = int(np.floor(midi_number / 12)) - 1  # Octave adjustment
    
    # Return the note (e.g., A4, C5, etc.)
    return f"{note_names[note_index]}{octave}"

# Example usage: Test the model with a single .wav file
file_path = 'C:/Users/hp/Downloads/clarinet.wav'  # Replace with the path to the wav file you want to test
test_wav_file(file_path)
