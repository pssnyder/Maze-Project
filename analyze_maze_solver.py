## Model Comparison Analysis Tool ##
#  This file takes in two .h5 models and outputs a summary of their Accuracy and Loss


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Nadam

# Load the models from the .h5 files
model1 = load_model('./models/maze_solver_20240719-091729.h5')
model2 = load_model('./models/.h5')

# Recreate and compile the optimizer for each model
model1.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
model2.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Generate example training data and labels
# Replace this with your actual data generation logic
num_samples = 1000
num_features = 6
num_classes = 4

training_data = np.random.rand(num_samples, num_features)
training_labels = np.eye(num_classes)[np.random.choice(num_classes, num_samples)]

# Fit the models and obtain the training history
history1 = model1.fit(training_data, training_labels, epochs=10, validation_split=0.2)
history2 = model2.fit(training_data, training_labels, epochs=10, validation_split=0.2)

# Extracting metrics from the training history
def extract_metrics(history):
    metrics = {
        'epoch': history.epoch,
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    return pd.DataFrame(metrics)

def plot_comparison(metrics1, metrics2):
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(metrics1['epoch'], metrics1['accuracy'], label='Run1 Accuracy')
    plt.plot(metrics2['epoch'], metrics2['accuracy'], label='Run2 Accuracy')
    plt.plot(metrics1['epoch'], metrics1['val_accuracy'], label='Run1 Val Accuracy', linestyle='dashed')
    plt.plot(metrics2['epoch'], metrics2['val_accuracy'], label='Run2 Val Accuracy', linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(metrics1['epoch'], metrics1['loss'], label='Run1 Loss')
    plt.plot(metrics2['epoch'], metrics2['loss'], label='Run2 Loss')
    plt.plot(metrics1['epoch'], metrics1['val_loss'], label='Run1 Val Loss', linestyle='dashed')
    plt.plot(metrics2['epoch'], metrics2['val_loss'], label='Run2 Val Loss', linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Extract metrics
metrics1 = extract_metrics(history1)
metrics2 = extract_metrics(history2)

# Compare metrics
comparison = pd.concat([metrics1, metrics2], axis=1, keys=['Run1', 'Run2'])
print(comparison)

# Plot the comparison
plot_comparison(metrics1, metrics2)