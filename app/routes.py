from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file,
    Response,
    jsonify
)
import zipfile
from zipfile import ZipFile
import os
import librosa
import joblib
import shutil
import numpy as np
import datetime
from app.progress import progress_info, progress_upload, progress_process, training_status
from app.ensembles import rf_training, xgb_training
from app.preprocessing import extract_zips, extract_features_labels, preprocess_features_labels
import threading
from pyngrok import ngrok


main = Blueprint("main", __name__)

PORT_NO = 5000
DEBUG = True
DOWNLOAD_MODEL_NAME = ""
public_url = ngrok.connect(PORT_NO).public_url


@main.route('/progress', methods=["GET"])
def progress():
    return jsonify(progress_info), 200


@main.route('/progress_process', methods=["GET"])
def progress_process_routes():
    return jsonify(progress_process), 200


@main.route("/progress_upload", methods=["GET"])
def progress_upload_routes():
    return jsonify(progress_upload), 200


@main.route("/")
def index():
    return render_template("layout.html")


@main.route("/recording", methods=["GET", "POST"])
def recording():
    number_of_classes = 3  # You can change this or make it dynamic
    if request.method == "POST":
        # Handle file uploads
        pass
    return render_template(
        "recording.html", number_of_classes=number_of_classes, debug=DEBUG
    )


@main.route("/delete_and_train", methods=["GET","POST"])
def delete_and_train():
    # Paths to the directories
    audio_samples_dir = os.path.join("data","audio_samples") 
    
    model_dir = os.path.join("data","model")
    # Reset the progress_upload dictionary
    progress_upload['progress'] = 0
    progress_upload['current_iteration'] = 0
    progress_upload['total_iteration'] = 0  # or whatever the default total_iteration value is

   

    progress_info['progress'] = 0.0
    progress_info['current_trial'] = 0
    progress_info['total_trials'] = 0  # or whatever the default total_iteration value is

    progress_process['progress'] = 0.0
    progress_process['current_trial'] = 0
    progress_process['total_trials'] = 0  # or whatever the default total_iteration value is
    # Delete all files in the specified directories
    for directory in [audio_samples_dir, model_dir]:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    return redirect(url_for('main.recording'))


@main.route("/training")
def training():
    return render_template("training.html", debug=DEBUG, progress=progress,)


# Function to load and resample audio files
def load_and_resample_audio(file_path, target_sampling_rate=16000):
    audio, _ = librosa.load(file_path, sr=target_sampling_rate)
    return audio


# Function to extract MFCC features
def extract_mfcc(audio, sr, n_mfcc=13):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)


@main.route("/train_model", methods=["POST"])
def train_model():
    # Directory where ZIP files were saved
    
    model_folder = os.path.join("data", "model", "")     
    reverse_class_mapping = session.get("reverse_class_mapping", {})
    
    upload_folder = os.path.join("data", "audio_samples", "")

    class_names = session.get("class_names", {})
    
    root_dir = os.path.join("data", "audio_samples")

    # Delete the model data
    
    delete_uploaded_data(model_folder)
    
    if not os.path.exists(os.path.dirname(model_folder)):
        os.makedirs(os.path.dirname(model_folder))

    # extract_zips(upload_folder)
    
    features, labels = extract_features_labels(root_dir, class_names)
    # print("Features",features)

    flattened_features, numerical_labels = preprocess_features_labels(features, labels)

    
    model_type = str(request.form["model"])
    
    n_trials = int((request.form["n_trials"]))

    if model_type == "random_forest":
        trained_model, best_params, class_report, plot_path = rf_training(flattened_features,numerical_labels,n_trials, reverse_class_mapping)
    elif model_type == "xgboost":
        trained_model, best_params, class_report, plot_path = xgb_training(flattened_features,numerical_labels,n_trials, reverse_class_mapping)

    #saved trained_model
    date_now = datetime.datetime.now().strftime(
        "%Y_%m_%d_%H_%M_%S"
    )  # Current date and time as a string in the format YYYY_MM_DD_HH_MM_SS

    extend_name =""
    for key, values in class_names.items():
        extend_name += ("_" + str(key))
    
    delete_uploaded_data(model_folder)
    
    if not os.path.exists(os.path.dirname(model_folder)):
        os.makedirs(os.path.dirname(model_folder))
    #model_path = r"data\model\trained_model_audio_{extend_name}.pkl".format(extend_name=extend_name)
    # OR using os.path.join for better compatibility
    model_path = os.path.join("data", "model", "trained_model_audio_{}.pkl".format(extend_name))
    #model_temp = "data/model/trained_model_audio.pkl"
    # OR using os.path.join for better compatibility
    model_temp = os.path.join("data", "model", "trained_model_audio.pkl")

    print("Model Path: ",model_path)
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))


    data_to_save = {
        "trained_model": trained_model,
        "class_names": class_names
    }

    if not os.path.exists(os.path.dirname(model_temp)):
        os.makedirs(os.path.dirname(model_temp))
    joblib.dump(data_to_save, model_path)
    joblib.dump(data_to_save, model_temp)
  
    
    
    # Delete the uploaded data
    upload_folder = "data/audio_samples/"
    delete_uploaded_data(upload_folder)
    print("Jession Class Name: ", class_names)
    if not os.path.exists(os.path.dirname(upload_folder)):
        os.makedirs(os.path.dirname(upload_folder))

   
    return render_template("training.html", class_report=class_report, plot_path=plot_path, training_complete=True)


@main.route("/prediction", methods=["GET"])
def prediction():
    return render_template("prediction.html", prediction=None)


@main.route("/upload_model_inference", methods=["GET"])
def upload_model_inference():
    return render_template("upload_model.html")


# Padding function
def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max([len(seq) for seq in sequences])
    padded_sequences = np.zeros((len(sequences), max_len, sequences[0].shape[1]))
    for i, seq in enumerate(sequences):
        padded_sequences[i, : len(seq)] = seq
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


@main.route("/predict", methods=["POST"])
def predict():
    # Load the trained model
   
    model_path = os.path.join("data", "model", "trained_model_audio.pkl")
    

    print("Jession Prediction: ", model_path)
    loaded_data = joblib.load(model_path)
    model = loaded_data["trained_model"]
    class_names = loaded_data["class_names"]

    # model = joblib.load(model_path)

    # Inverse label mapping based on the session's class names
    # class_names = session.get("class_names", {})
    print("Jession Predicion: ", class_names)
    inverse_label_mapping = {v: k for k, v in class_names.items()}

    # Load the uploaded audio file
    uploaded_file = request.files["audio_file"]
    if uploaded_file and uploaded_file.filename.endswith(".wav"):
        # Save the uploaded file to a temporary location
       
        temp_path = os.path.join("data", "temp_audio.wav")

        uploaded_file.save(temp_path)
        print(temp_path)
        # Predict the class using the original routine
        predicted_class_name = predict_audio_class(
            temp_path, model, inverse_label_mapping
        )

        # Render the result
        return render_template("prediction.html", prediction=predicted_class_name)
    return redirect(
        url_for("main.prediction")
    )  # Redirect back if no valid file uploaded

@main.route("/save_model", methods=["POST"])
def save_model():
    # Load the trained model
    
    
    class_names = session.get("class_names", {})

    extend_name =""
    for key, values in class_names.items():
        extend_name += ("_" + str(key))

    # Load the uploaded audio file
    uploaded_file = request.files["pickle_file"]

    if uploaded_file and uploaded_file.filename.endswith(".pkl"):
        # Save the uploaded file to a temporary location
        
        temp_path = os.path.join("data", "model", "trained_model_audio_{extend_name}.pkl".format(extend_name=extend_name))

        uploaded_file.save(temp_path)
        print(temp_path)
        # Predict the class using the original routine
        
    return redirect(
        url_for("main.prediction")
    )  # Redirect back if no valid file uploaded


@main.route("/download_model")
def download_model():

    class_names = session.get("class_names", {})

    extend_name =""
    for key, values in class_names.items():
        extend_name += ("_" + str(key))
    
    
    model_path = os.path.join("data", "model", "trained_model_audio_{extend_name}.pkl".format(extend_name=extend_name))
    model_filename = os.path.basename(model_path)  # Get the filename from the path

    with open(model_path, "rb") as file:
        content = file.read()

    response = Response(
        content,
        headers={"Content-Disposition": f"attachment; filename={model_filename}"},
    )
    response.mimetype = "application/octet-stream"

    return response




@main.route("/upload", methods=["POST"])
def upload():
    print(request.files.keys())
    print(request.form.keys())

    total_iterations = len(request.files.keys()) 
    
    upload_folder = os.path.join("data", "audio_samples", "")


    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    class_names = {}
    i = 0
    while request.form.get(f"class_name{i}"):
        
        class_name = request.form[f"class_name{i}"].replace(".zip", "").lower()
        class_names[class_name] = i
        class_file = request.files[
            f"class{i}"
        ]  # This line should match the HTML input name
        if class_file and class_file.filename.endswith(".zip"):
            zip_path = os.path.join(upload_folder, f"{class_name}.zip")
            class_file.save(zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(upload_folder, class_name))
        
        progress_upload['progress'] = ((i+1)/ total_iterations) * 100
        progress_upload['current_iteration'] = i+1
        progress_upload['total_iteration'] = total_iterations
        i += 1
    reverse_class_mapping = {v: k for k, v in class_names.items()}
    session["class_names"] = class_names
    session["reverse_class_mapping"] = reverse_class_mapping

    return redirect(url_for("main.training"))


def delete_uploaded_data(directory_path):
    try:
        shutil.rmtree(directory_path)
        print("Uploaded data deleted successfully.")
    except Exception as e:
        print(f"An error occurred while deleting uploaded data: {str(e)}")

