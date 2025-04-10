import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model
model = load_model("fmri_brain_model.keras")

# Create the GUI app
app = tk.Tk()
app.title("fMRI Brain Region Classifier & Signal Analyzer")

# Create input field for signals
input_label = tk.Label(app, text="Enter Signals (comma-separated):")
input_label.pack()

input_field = tk.Entry(app, width=50)
input_field.pack()

# Function to process input, make predictions, and analyze signals
def predict():
    try:
        # Get input data and process it
        signals = input_field.get()
        signals_list = [float(i.strip()) for i in signals.split(",")]
        signals_array = np.array(signals_list).reshape(-1, 1)  # Reshape for the model
        
        # Make predictions using the model
        predictions = model.predict(signals_array)
        predicted_labels = np.argmax(predictions, axis=1)
        confidence_scores = [f"{np.max(pred) * 100:.2f}%" for pred in predictions]
        probabilities = [pred.tolist() for pred in predictions]  # Full probability distribution
        
        # Define thresholds for anomaly detection (adjust based on dataset analysis)
        normal_range = (-0.1, 0.1)
        
        # Generate insights
        insights = []
        for i, (signal, label, confidence, prob_dist) in enumerate(zip(signals_list, predicted_labels, confidence_scores, probabilities)):
            # Anomaly detection
            is_anomalous = signal < normal_range[0] or signal > normal_range[1]
            condition = "Normal"
            if is_anomalous:
                # Example condition association (adjust based on research/data)
                condition = "Possible indicator of Alzheimer's" if label == 0 else "Potential anomaly in brain function"
            
            insights.append(
                f"Signal {i + 1}: {signal}\n"
                f" - Predicted Region: {label}\n"
                f" - Confidence: {confidence}\n"
                f" - Probability Distribution: {prob_dist}\n"
                f" - Anomaly Detected: {'Yes' if is_anomalous else 'No'}\n"
                f" - Associated Condition: {condition}\n"
            )
        
        # Show the insights in a message box
        result = "\n\n".join(insights)
        messagebox.showinfo("Prediction Results & Insights", result)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create a button to trigger predictions
predict_button = tk.Button(app, text="Predict", command=predict)
predict_button.pack()

# Run the GUI app
app.mainloop()
