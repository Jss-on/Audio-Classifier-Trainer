# preprocessing.py

from zipfile import ZipFile
import os
import librosa
import numpy as np
from app.progress import progress_process

# Function to load and resample audio files
def load_and_resample_audio(file_path, target_sampling_rate=16000):
    audio, _ = librosa.load(file_path, sr=target_sampling_rate)
    return audio


# Function to extract MFCC features
def extract_mfcc(audio, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


# Padding function
def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])
    padded_sequences = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq
    return padded_sequences


def extract_zips(upload_folder):
    extracted_paths = {}
    for class_name in os.listdir(upload_folder):
        class_name = class_name.replace(".zip", "").lower()  # Convert to lower case
        zip_path = os.path.join(upload_folder, f"{class_name}.zip")
        extract_path = os.path.join(upload_folder, class_name)
        extracted_paths[class_name] = extract_path
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
    return extracted_paths

def extract_features_labels(root_dir, class_names):
    features = []
    labels = []

    number_of_files = 0

    for root, dirs, files in os.walk(root_dir):
        number_of_files += len(files)

    i = 0
    for root, dirs, files in os.walk(root_dir):
        # Skip the main directory itself
        
        if root == root_dir:
            continue
        # Check if nested directory
        if len(root.split(os.sep)) == len(root_dir.split(os.sep)) + 2:
            class_name = os.path.basename(os.path.dirname(root)).lower()
        else:
            class_name = os.path.basename(root).lower()
        print("files", files)
        for file_name in files:
            if file_name.endswith(".wav"):
                file_path = os.path.join(root, file_name)
                print(file_path)
                audio = load_and_resample_audio(file_path)
                mfcc_features = extract_mfcc(audio, sr=16000)
                features.append(mfcc_features.T)
                labels.append(class_names[class_name])
                progress_process["progress"] = (i/number_of_files) * 100
                progress_process["current_iteration"] = i + 1
                progress_process["total_iteration"] = number_of_files
                
                i += 1

    return features, labels

def preprocess_features_labels(features, labels):
    padded_features = pad_sequences(features)
    flattened_features = padded_features.reshape(padded_features.shape[0], -1)
    numerical_labels = labels  # Correct mapping

    return flattened_features, numerical_labels
