import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------
# **1. Manually Select Dataset**
# -------------------------------
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select fMRI Dataset", filetypes=[("NPZ files", "*.npz")])
if not file_path:
    raise FileNotFoundError("No dataset selected.")

data = np.load(file_path, allow_pickle=True)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# -------------------------------
# **2. Preprocess Data**
# -------------------------------
num_classes = len(np.unique(y_train))
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# -------------------------------
# **3. Define Advanced Model**
# -------------------------------
model = Sequential([
    Dense(1024, activation="relu", kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

# Optimizer with weight decay
initial_learning_rate = 0.001
lr_scheduler = LearningRateScheduler(lambda epoch: initial_learning_rate * 0.95 ** epoch)

model.compile(optimizer=AdamW(learning_rate=initial_learning_rate, weight_decay=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# -------------------------------
# **4. Train Model with Advanced Settings**
# -------------------------------
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test),
                    epochs=150,  # Increased epochs for better convergence
                    batch_size=32,
                    verbose=1,
                    callbacks=[early_stopping, lr_scheduler])

# -------------------------------
# **5. Save Model**
# -------------------------------
model.save("advanced_fmri_model.keras")
print("✅ Model saved as: advanced_fmri_model.keras")

# -------------------------------
# **6. Evaluate Model**
# -------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=32)
print(f"🎯 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}")

# -------------------------------
# **7. Plot Training Results**
# -------------------------------
def plot_training_results(history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.title("Model Accuracy")
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Model Loss")
    
    plt.show()

plot_training_results(history)
