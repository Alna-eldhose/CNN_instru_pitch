# CNN_instru_pitch
Overview
This project applies deep learning to classify musical instruments and recognize pitch from audio recordings. The model processes spectrograms from audio files and uses a Convolutional Neural Network (CNN) to achieve high accuracy in instrument and pitch classification.

Dataset
Source: TinySOL dataset (Ircam, Studio On Line - SOL)https://zenodo.org/records/3685367#.Xo1NVi2ZOuU
Total Samples: 2913 audio WAV files (44.1 kHz)
Instruments Included (14 classes):
Trumpet
Alto Saxophone
Violin
Contrabass
Flute, Oboe, Clarinet, Bassoon
Trombone, French Horn, Tuba, Accordion, Cello, Viola  

Preprocessing
Converted raw WAV files to Mel-Spectrograms (log-scaled).
Applied Min-Max Normalization for CNN input.
Data augmentation (time-shifting, noise addition) to improve generalization.
Model Architecture
The CNN model consists of:

Feature Extraction Layers:
3 Convolutional Layers with ReLU activation
2 Max Pooling Layers for downsampling
Classification Layers:
Flatten Layer
2 Fully Connected (Dense) Layers
Softmax Output Layer for multi-class classification
Loss Function: categorical_crossentropy
Optimizer: Adam
Evaluation Metrics: Accuracy, Loss, Confusion Matrix

Training & Performance
Instrument Recognition Model:

Accuracy: 91% (32 epochs)
Loss: Low (consistent improvement across epochs)
Pitch Recognition Model:

Accuracy: 85% (9 epochs)
Loss: Higher than the instrument model due to pitch complexity
Train-Test Split: 80% Training, 20% Testing

Results & Evaluation
Accuracy & Loss Graphs (included in the repository)
Confusion Matrix for classification performance analysis
Spectrogram Analysis showing real-time instrument feature extraction
