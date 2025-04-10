import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import filedialog, messagebox

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------
# 1. Load Trained Model
# -------------------------------
model_path = "fmri_brain_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Trained model not found at: {model_path}")

model = load_model(model_path)
print(f"✅ Loaded model from {model_path}")

# -------------------------------
# 2. Load Test Dataset
# -------------------------------
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Test fMRI Dataset", filetypes=[("NPZ files", "*.npz")])
if not file_path:
    raise FileNotFoundError("No dataset selected.")

try:
    data = np.load(file_path, allow_pickle=True)
    print(f"Available keys in NPZ file: {list(data.keys())}")
except Exception as e:
    raise ValueError(f"Error loading the dataset: {e}")

X_test = data["X_test"]
y_test = data["y_test"]

# -------------------------------
# 3. Preprocess Data
# -------------------------------
X_test = X_test.reshape(X_test.shape[0], -1)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

num_classes = len(np.unique(y_test))
y_test_cat = to_categorical(y_test, num_classes)

# -------------------------------
# 4. Evaluate Model
# -------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, batch_size=32)
print(f"🎯 Test Loss: {test_loss}")
print(f"🎯 Test Accuracy: {test_accuracy}")

# -------------------------------
# 5. Show Accuracy in a New Tkinter Window
# -------------------------------
def show_accuracy_window():
    acc_window = tk.Toplevel()
    acc_window.title("Test Accuracy Report")
    acc_window.geometry("300x150")
    msg = f"🎯 Test Accuracy: {test_accuracy*100:.2f}%\n📉 Test Loss: {test_loss:.4f}"
    label = tk.Label(acc_window, text=msg, font=("Helvetica", 14))
    label.pack(pady=30)

    # Plot accuracy as a bar
    fig, ax = plt.subplots()
    ax.bar(["Accuracy"], [test_accuracy * 100], color="skyblue")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Model Test Accuracy")
    plt.tight_layout()
    plt.show()

show_accuracy_window()

# -------------------------------
# 6. Predict & Display Results
# -------------------------------
predictions = model.predict(X_test[:5])
predicted_labels = np.argmax(predictions, axis=1)
actual_labels = np.argmax(y_test_cat[:5], axis=1)

for i in range(5):
    confidence = np.max(predictions[i]) * 100
    print(f"\n🔍 Prediction {i + 1}: {predicted_labels[i]} (Confidence: {confidence:.2f}%)")
    print(f"🎯 Actual Label {i + 1}: {actual_labels[i]}")

# -------------------------------
# 7. Show Prediction Graph in Separate Window
# -------------------------------
def show_prediction_plot():
    plt.figure(figsize=(10, 5))
    bar_width = 0.1
    indices = np.arange(num_classes)
    
    for i in range(5):
        plt.bar(indices + i * (bar_width + 0.02), predictions[i], width=bar_width, label=f"Sample {i+1}")
    
    plt.xticks(indices + 0.2, [f"Class {i}" for i in range(num_classes)])
    plt.title("Predicted Class Probabilities (First 5 Samples)")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

show_prediction_plot()
