import numpy as np
import pyvista as pv
import nibabel as nib
from nilearn import datasets
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("fmri_brain_model.keras")
expected_shape = model.input_shape[1]

# Load fsaverage brain mesh (left hemisphere)
fsaverage = datasets.fetch_surf_fsaverage()
gii = nib.load(fsaverage['pial_left'])
coords = gii.darrays[0].data
faces = gii.darrays[1].data
faces = np.column_stack((np.full(len(faces), 3), faces))  # triangle format
brain_mesh = pv.PolyData(coords, faces)

# Load Destrieux atlas
atlas = datasets.fetch_atlas_surf_destrieux()
labels = atlas["map_left"]
names = atlas["labels"]

# Create label mapping from index to region name
label_mapping = {i: name.decode('utf-8') if isinstance(name, bytes) else name for i, name in enumerate(names)}

# GUI Setup
app = tk.Tk()
app.title("fMRI Brain Region Classifier")

tk.Label(app, text="Enter fMRI Signals (comma-separated):").pack()
input_field = tk.Entry(app, width=60)
input_field.pack()

def predict_and_visualize():
    try:
        raw = input_field.get().strip()
        values = [float(i) for i in raw.split(",")]

        if len(values) != expected_shape:
            messagebox.showerror("Shape Error", f"Expected {expected_shape} values, got {len(values)}")
            return

        input_array = np.array(values).reshape(1, expected_shape)
        prediction = model.predict(input_array)
        predicted_label_index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100

        region_name = label_mapping.get(predicted_label_index, "Unknown")

        messagebox.showinfo("Prediction", f"Predicted Region: {region_name}\nConfidence: {confidence:.2f}%")
        show_brain(predicted_label_index)

    except Exception as e:
        messagebox.showerror("Error", str(e))

def show_brain(label_index):
    plotter = pv.Plotter()
    scalars = np.array([1 if val == label_index else 0 for val in labels])
    plotter.add_mesh(brain_mesh, scalars=scalars, cmap="coolwarm", opacity=0.85, show_scalar_bar=False)
    plotter.add_title(label_mapping.get(label_index, "Unknown Region"))
    plotter.show()

tk.Button(app, text="Predict & Visualize", command=predict_and_visualize).pack()
app.mainloop()
