# ensembles.py

import optuna
from optuna.samplers import TPESampler
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from app.progress import progress_info
import threading
import pandas as pd

# # The flag to signal the training to stop
# should_stop = threading.Event()



def objective_rf(trial, n_trials, flattened_features, numerical_labels):
    # defining parameters
    params = {
        "test_size": trial.suggest_uniform("test_size", 0.1, 0.4),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 10, 100),
        "criterion": trial.suggest_categorical('criterion', ['gini', 'entropy'])
        
    }

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        flattened_features,
        numerical_labels,
        test_size=params["test_size"],
        random_state=42,
        stratify=numerical_labels,
    )
    
    # Remove 'test_size' as it is not a valid parameter for RandomForestClassifier
    del params['test_size']
    
    # Initializing the Random Forest model
    model = RandomForestClassifier(**params, oob_score=True, n_jobs=-1, random_state=42)

    # Fitting the model
    model.fit(X_train, y_train)

    # Validate
    score = model.score(X_test, y_test)
    # If the score is 1.0, set the stop flag to stop training
    
    progress_info['progress'] = (trial.number / n_trials) * 100  # This gives progress as a percentage
    progress_info['current_trial'] = trial.number
    progress_info['total_trials'] = n_trials
    return score


def rf_training(flattened_features, numerical_labels, n_trials, reverse_class_mapping):
    return train_model_with_optuna(flattened_features, numerical_labels, n_trials, objective_rf, reverse_class_mapping)

# define objective function for xgboost
def objective_xgb(trial, n_trials, flattened_features, numerical_labels):
    # define parameters for xgboost model
    test_size = trial.suggest_uniform("test_size", 0.1, 0.4)
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 10, 100)
    learning_rate = trial.suggest_uniform("learning_rate", 0.01, 1)
    
    classifier = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        flattened_features,
        numerical_labels,
        test_size=test_size,
        random_state=42,
        stratify=numerical_labels,
    )

    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)

    # If the score is 1.0, set the stop flag to stop training
    progress_info['progress'] = (trial.number / n_trials) * 100  # This gives progress as a percentage
    progress_info['current_trial'] = trial.number
    progress_info['total_trials'] = n_trials
    print(score)
    return score

def xgb_training(flattened_features, numerical_labels, n_trials, reverse_class_mapping):
    return train_model_with_optuna(flattened_features, numerical_labels, n_trials, objective_xgb, reverse_class_mapping)
def parse_classification_report(report):
    """Parse the classification report into a dictionary."""
    report_data = []
    lines = report.split('\n')
    
    for line in lines[2:-3]:
        row_data = line.strip().split()
        if len(row_data) != 5:  # Check if the row format is correct
            continue
        try:
            # Check if the first value in row is not a string, if so skip the row
            float(row_data[1])
            report_data.append({
                'class': row_data[0],
                'precision': float(row_data[1]),
                'recall': float(row_data[2]),
                'f1-score': float(row_data[3]),
                'support': float(row_data[4])
            })
        except ValueError:
            continue

    return report_data

def train_model_with_optuna(flattened_features, numerical_labels, n_trials, objective_func, reverse_class_mapping):
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    study.optimize(partial(objective_func, n_trials=n_trials, 
                            flattened_features=flattened_features, 
                            numerical_labels=numerical_labels), n_trials=n_trials)
    
    best_params = study.best_params

    X_train, X_test, y_train, y_test = train_test_split(
        flattened_features,
        numerical_labels,
        test_size=best_params["test_size"],
        random_state=42,
        stratify=numerical_labels,
    )

    del best_params["test_size"]
    if objective_func == objective_rf:
        best_model = RandomForestClassifier(**best_params, random_state=42)
    elif objective_func == objective_xgb:
        best_model = XGBClassifier(**best_params, random_state=42)

    best_model.fit(X_train, y_train)

     # Generate classification report
    y_pred = best_model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=list(reverse_class_mapping.values()))


    
    # Generate confusion matrix
    y_test_names = [reverse_class_mapping[str(label)] for label in y_test]
    predicted_names = [reverse_class_mapping[str(label)] for label in best_model.predict(X_test)]
    accuracy = accuracy_score(y_test, best_model.predict(X_test))
    cm = confusion_matrix(y_test_names, predicted_names, labels=list(reverse_class_mapping.values()))

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=list(reverse_class_mapping.values()),
        yticklabels=list(reverse_class_mapping.values()),
    )

    # Set the labels for the x and y axes
    ax.set_xlabel("Predicted", color="red", weight="bold")
    ax.set_ylabel("Actual", color="red", weight="bold")
    ax.set_title(f"Confusion Matrix (Accuracy: {accuracy*100:.2f}%)")  # Added accuracy to the title

    plot_path = "app/static/plot.png"
    fig.savefig(plot_path)
    plt.close(fig)

    
    return best_model, best_params, class_report, plot_path
