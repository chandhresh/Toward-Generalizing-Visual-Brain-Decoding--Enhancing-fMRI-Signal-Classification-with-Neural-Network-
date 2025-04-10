 🧠 Brain Decoding from fMRI Signals using Deep Learning

This project implements a deep learning approach to decode brain activity from fMRI signals. It provides a GUI-based classifier and a 3D brain visualization tool to highlight active regions. The goal is to classify brain states in real time and support early detection of neurological conditions like Alzheimer’s disease.

📁 Project Structure

| File/Folder                | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `app.py`                  | **Main GUI** for loading fMRI input and displaying predictions              |
| `3d1.py`                  | **3D brain visualization** to highlight predicted brain regions             |
| `train.py`, `train2.py`   | Scripts for training models on fMRI datasets                                |
| `fmri_data.npz`           | Preprocessed fMRI dataset with labeled brain regions                        |
| `*.keras` files           | Saved models (`advanced_fmri_model.keras`, `fmri_brain_model.keras`, etc.)  |
| `output1.png`, `output2.png` | Example output or sample GUI screenshots                                 |
| `test.py`, `fmri_app.py`  | Miscellaneous or legacy testing and GUI experiments                         |

🧪 Model Overview

- Framework: **TensorFlow/Keras**
- Model Type: **Sequential Neural Network**
- Layers: Dense, Dropout, Batch Normalization
- Training: Supervised learning on labeled fMRI regions

---
 📊 Dataset Details

- File: `fmri_data.npz`
- Format: NumPy array with fMRI signals and labels
- Preprocessed for training and inference (standardized and reshaped)
- Must be placed in the same directory as `app.py` during execution


 🚀 How to Run
 1. Clone the repository:

```bash
git clone https://github.com/your-username/brain-decoding-fmri.git
cd brain-decoding-fmri
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

<details>
<summary>📌 Recommended Libraries (if requirements.txt is missing)</summary>

```bash
pip install numpy tensorflow scikit-learn matplotlib nibabel nilearn tkinter
```

</details>
 3. Launch the main GUI (`app.py`):

```bash
sreamlit run app.py

You can now upload an fMRI file and get real-time predictions about brain region activity.

 🧠 3D Brain Visualization (`3d1.py`)

This script uses the prediction output from the model to generate a **3D visualization of the brain**. It highlights the region associated with the prediction on a simulated 3D brain model.

 How to Use:

```bash
python 3d1.py
```

- Ensure the output from `app.py` is stored or accessible.
- You may need to manually enter the predicted label if prompted.
- A 3D rendering of the brain will appear with the highlighted region.

 📦 Saved Models

You can choose from three pretrained models included:

- `fmri_brain_model.keras`
- `advanced_fmri_model.keras`
- `improved_fmri_model.keras`

These can be loaded in `app.py` by modifying the model loading section depending on which model you want to test.
 📸 Sample Outputs

Included in the project:
- `output1.png`: Sample GUI interface with prediction
- `output2.png`: Brain activity classification display

Perfect! Since you're using the **Seaborn fMRI dataset**, I'll include a clear section in the README to describe it accurately.

Here’s the updated **Dataset Details** section you can include in your `README.md`:

---

 📊 Dataset Details: Seaborn fMRI Dataset

- **Source**: Built-in dataset from the [Seaborn](https://seaborn.pydata.org/) visualization library.
- **Description**: The dataset contains simulated fMRI measurements from subjects who performed tasks under different conditions. It’s structured for exploratory data analysis and machine learning applications.
- **Fields Include**:
  - `subject`: ID of the subject
  - `region`: Brain region (e.g., frontal, parietal)
  - `event`: Task condition (e.g., "stim", "cue")
  - `signal`: fMRI signal response (numerical)
- **Use in This Project**:
  - The dataset is **preprocessed and converted into NumPy format** (`fmri_data.npz`) for deep learning model training.
  - Signals are normalized and reshaped to fit the input structure required by the neural network.

 Example from Raw Dataset:
```plaintext
subject   region   event   signal
-------------------------------------
s13       frontal  stim    0.265
s13       frontal  stim    0.252
s13       parietal stim    0.234
...
```

 Format in Project:
```python
npz_data = np.load("fmri_data.npz")
X = npz_data['X']  # Preprocessed signal data
y = npz_data['y']  # Labels (brain regions or events)
