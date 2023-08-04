from flask import Blueprint, render_template, request, redirect, url_for, session, send_file, Response
import zipfile
from zipfile import ZipFile
import os
import librosa
import joblib
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('layout.html')

@main.route('/recording', methods=['GET', 'POST'])
def recording():
    number_of_classes = 3  # You can change this or make it dynamic
    if request.method == 'POST':
        # Handle file uploads
        pass
    return render_template('recording.html', number_of_classes=number_of_classes)

@main.route('/training')
def training():
    return render_template('training.html')

# Function to load and resample audio files
def load_and_resample_audio(file_path, target_sampling_rate=16000):
    audio, _ = librosa.load(file_path, sr=target_sampling_rate)
    return audio

# Function to extract MFCC features
def extract_mfcc(audio, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

@main.route('/train_model', methods=['POST'])
def train_model():
    # Directory where ZIP files were saved
    upload_folder = 'data/audio_samples/'

     # Retrieve class names from the session
    class_names = session.get('class_names', {})
    reverse_class_mapping = session.get('reverse_class_mapping', {})
    # print("Jessopn",reverse_class_mapping)
    # Extracting ZIP files
    extracted_paths = {}
    for class_name in os.listdir(upload_folder):
        class_name = class_name.replace('.zip', '').lower()  # Convert to lower case
        zip_path = os.path.join(upload_folder, f'{class_name}.zip')
        extract_path = os.path.join(upload_folder, class_name)
        extracted_paths[class_name] = extract_path
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # # Function to load and resample audio files
    # def load_and_resample_audio(file_path, target_sampling_rate=16000):
    #     audio, _ = librosa.load(file_path, sr=target_sampling_rate)
    #     return audio

    # # Function to extract MFCC features
    # def extract_mfcc(audio, sr, n_mfcc=13):
    #     return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


    # Collecting features and labels
    features = []
    labels = []
    for label, path in extracted_paths.items():
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            audio = load_and_resample_audio(file_path)
            mfcc_features = extract_mfcc(audio, sr=16000)
            features.append(mfcc_features.T)
            labels.append(class_names[label])  # Map labels to numerical values

    # Padding, flattening, and mapping the features and labels
    # max_len = max([len(feature) for feature in features])
    padded_features = pad_sequences(features)
    flattened_features = padded_features.reshape(padded_features.shape[0], -1)
    numerical_labels = labels  # Correct mapping

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(flattened_features, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels)

    # Initializing the Random Forest model
    n_estimators = int(request.form['n_estimators'])
    

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Fitting the model
    model.fit(X_train, y_train)
    # Save the trained model

    model_path = 'data/model/random_forest_model.pkl'
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    joblib.dump(model, model_path)

    # Evaluating the model
    accuracy = accuracy_score(y_test, model.predict(X_test))
   
    y_test_names = [reverse_class_mapping[str(label)] for label in y_test]
    predicted_names = [reverse_class_mapping[str(label)] for label in model.predict(X_test)]
    cm = confusion_matrix(y_test_names, predicted_names, labels=list(reverse_class_mapping.values()))
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=list(reverse_class_mapping.values()), yticklabels=list(reverse_class_mapping.values()))

    plot_path = 'app/static/plot.png'
    fig.savefig(plot_path)
    plt.close(fig)

    # Delete the uploaded data
    upload_folder = 'data/audio_samples/'
    delete_uploaded_data(upload_folder)
    
    return render_template('training.html', plot_path=plot_path, training_complete=True)

@main.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html', prediction=None)

# Padding function
def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])
    padded_sequences = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences
# Function to predict audio class (same as original)
def predict_audio_class(file_path, model, label_mapping):
    # Creating an inverse label mapping
    # inverse_label_mapping = {v: k for k, v in label_mapping.items()}
    # Loading and resampling the audio file
    audio = load_and_resample_audio(file_path)
    # Extracting MFCC features
    mfcc = extract_mfcc(audio, sr=16000)
    # Padding the features
    # max_len = 170
    # padded_features = np.pad(mfcc, pad_width=((0, 0), (0, max_len - len(mfcc))), mode='constant')
    padded_features = pad_sequences([mfcc.T])
    # Flattening the features
    flattened_features = padded_features.flatten().reshape(1, -1)
    print(flattened_features)
    # Making a prediction using the trained model
    prediction = model.predict(flattened_features)
    
    predicted_class = label_mapping[prediction[0]]
    return predicted_class 

@main.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    model_path = 'data/model/random_forest_model.pkl'
    model = joblib.load(model_path)

    # Inverse label mapping based on the session's class names
    class_names = session.get('class_names', {})
    inverse_label_mapping = {v: k for k, v in class_names.items()}

    # Load the uploaded audio file
    uploaded_file = request.files['audio_file']
    if uploaded_file and uploaded_file.filename.endswith('.wav'):
        # Save the uploaded file to a temporary location
        temp_path = 'data/temp_audio.wav'
        uploaded_file.save(temp_path)
        print(temp_path)
        # Predict the class using the original routine
        predicted_class_name = predict_audio_class(temp_path, model, inverse_label_mapping)

        # Render the result
        return render_template('prediction.html', prediction=predicted_class_name)

    return redirect(url_for('main.prediction'))  # Redirect back if no valid file uploaded


@main.route('/download_model')
def download_model():
    model_path = 'data/model/random_forest_model.pkl'
    with open(model_path, 'rb') as file:
        content = file.read()
    response = Response(content, headers={'Content-Disposition': 'attachment; filename=random_forest_model.pkl'})
    response.mimetype = 'application/octet-stream'
    return response

@main.route('/upload', methods=['POST'])
def upload():
    upload_folder = 'data/audio_samples/'

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    class_names = {}
    i = 0
    while request.form.get(f'class_name{i}'):
        class_name = request.form[f'class_name{i}'].replace('.zip', '').lower()
        class_names[class_name] = i
        class_file = request.files[f'class{i}']
        if class_file and class_file.filename.endswith('.zip'):
            zip_path = os.path.join(upload_folder, f'{class_name}.zip')
            class_file.save(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(upload_folder, class_name))
        i += 1

    reverse_class_mapping = {v: k for k, v in class_names.items()}
    session['class_names'] = class_names
    session['reverse_class_mapping'] = reverse_class_mapping

    return redirect(url_for('main.training'))


def delete_uploaded_data(directory_path):
    try:
        shutil.rmtree(directory_path)
        print("Uploaded data deleted successfully.")
    except Exception as e:
        print(f"An error occurred while deleting uploaded data: {str(e)}")
