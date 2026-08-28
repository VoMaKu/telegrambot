from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np
import librosa
import os

# Load dataset

dataset = "dataset/splitted_final/"
num_labels = 10

labels = []
audios = []
for label in range(num_labels):
    label_path = f"{dataset}/{label}"
    for file in sorted(os.listdir(label_path)):
        file_path = label_path + "/" + file
        sample_rate, audio = read(file_path)
        labels.append(label)
        audios.append(audio)
labels = np.array(labels)

# Prepare features

max_duration_sec = 0.8
max_duration = int(max_duration_sec * sample_rate + 1e-6)

features = []
features_flatten = []
for audio in audios:
    if len(audio) < max_duration:
        audio = np.pad(audio, (0, max_duration - len(audio)), constant_values=0)
    feature = librosa.feature.melspectrogram(y=audio.astype(float), sr=sample_rate, n_mels=32, fmax=4096)
    features.append(feature)
    features_flatten.append(feature.reshape(-1))

#print([feature.shape for feature in features])

def plot(idx):
    """Show the waveform and the mel-spectrogram of one sample (notebook only)."""
    from IPython.display import Audio, display

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"{labels[idx]}")
    plt.plot(audios[idx])

    plt.subplot(1, 2, 2)
    plt.title(f"{labels[idx]}")
    plt.imshow(features[idx])

    display(Audio(audios[idx], rate=sample_rate))

features_train, features_test, labels_train, labels_test = train_test_split(features_flatten, labels)

# model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
model = MLPClassifier(hidden_layer_sizes=(888, 777, 666))

model.fit(X=features_train, y=labels_train)

# Save model

import pickle
filename = "model.pkl"
model_pickled = pickle.dumps(model)
with open(filename, 'wb') as f:
    f.write(model_pickled)

# Load model


import pickle
filename = "model.pkl"
with open(filename, 'rb') as f:
    model_pickled = f.read()
model = pickle.loads(model_pickled)

# Validate model

labels_test_predicted = model.predict(X=features_test)

(labels_test_predicted == labels_test).mean()

print((labels_test_predicted == labels_test).mean())

print(labels_test_predicted)

print(labels_test)