import os
import pickle
import sys

import librosa
import numpy as np
from scipy.io.wavfile import read

# These must match the values training.py used, otherwise the flattened feature
# vector has a different length than the one the model was fitted on.
MAX_DURATION_SEC = 0.8
N_MELS = 32
FMAX = 4096
# Both bots convert every voice message to this rate before anything else, so the
# model has only ever seen 48 kHz audio. The number of mel frames -- and with it
# the length of the feature vector -- follows from the rate, so a file recorded at
# any other rate produces a vector the model cannot accept.
SAMPLE_RATE = 48000


def extract_features(audio, sample_rate):
    """Turn one recording into the flat feature vector the model expects.

    Args:
        audio: samples of a single spoken digit, already cut out by the VAD.
        sample_rate: sample rate the recording was read with.

    Returns:
        A 1-D numpy array of mel-spectrogram values.
    """
    max_duration = int(MAX_DURATION_SEC * sample_rate + 1e-6)
    # The model was fitted on a fixed length, so shorter clips are padded and
    # longer ones cut, otherwise the vector below would not match.
    if len(audio) < max_duration:
        audio = np.pad(audio, (0, max_duration - len(audio)), constant_values=0)
    else:
        audio = audio[:max_duration]
    feature = librosa.feature.melspectrogram(
        y=audio.astype(float), sr=sample_rate, n_mels=N_MELS, fmax=FMAX)
    return feature.reshape(-1)


def load_model(filename="model.pkl"):
    """Unpickle the classifier trained by training.py."""
    if not os.path.exists(filename):
        print(f"{filename} not found. Run 'make training' to build it from dataset/splitted_final.")
        exit(1)
    with open(filename, "rb") as f:
        return pickle.loads(f.read())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect args. Example:")
        print("python3 inference.py inference/splitted/unk.wav")
        exit(1)

    wav_file_path = sys.argv[1]
    sample_rate, audio = read(wav_file_path)
    if sample_rate != SAMPLE_RATE:
        print(f"{wav_file_path}: expected {SAMPLE_RATE} Hz, got {sample_rate} Hz.")
        print(f"Convert it first: ffmpeg -i {wav_file_path} -ar {SAMPLE_RATE} -ac 1 out.wav")
        exit(1)

    model = load_model()
    answer = model.predict([extract_features(audio, sample_rate)])[0]
    print(answer)
