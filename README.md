# Face Expression Recognition using CNN

This repository contains a Deep Learning project that builds a Convolutional Neural Network (CNN) to classify human facial expressions into distinct emotion categories.

---

## 📌 Project Overview
The model processes grayscale facial images and predicts the underlying emotion. It handles 7 distinct emotion classes: **angry, disgust, fear, happy, neutral, sad, and surprise**. The final trained model is exported and saved as `emotion_detection.h5` for future inference.

## 📊 Dataset and Preprocessing
The model expects image inputs to be resized to **48x48 pixels in grayscale** (`color_mode='grayscale'`). 
To improve model generalization and prevent overfitting, the training data is augmented using `ImageDataGenerator` with the following parameters:
* **Rescaling**: 1./255
* **Rotation Range**: 20 degrees
* **Zoom Range**: 20% (0.2)
* **Horizontal Flip**: True

Validation data is only rescaled (1./255) without any augmentations to ensure objective evaluation.

---

## 🧠 Model Architecture
The neural network is constructed using the Keras `Sequential` API. The architecture consists of multiple convolutional blocks followed by fully connected layers:
1. **Block 1**: `Conv2D` (64 filters, 3x3) > `BatchNormalization` > `MaxPooling2D` (2x2) > `Dropout` (0.25).
2. **Block 2**: `Conv2D` (128 filters, 3x3) > `BatchNormalization` > `MaxPooling2D` (2x2) > `Dropout` (0.25).
3. **Block 3**: `Conv2D` (256 filters, 3x3) > `BatchNormalization` > `MaxPooling2D` (2x2) > `Dropout` (0.25).
4. **Classification Head**: `Flatten` > `Dense` (512 units) > `BatchNormalization` > `Dropout` (0.5) > `Dense` (7 units, Softmax activation).

---

## ⚙️ Training details
* **Optimizer**: Adam with an initial learning rate of 0.001.
* **Loss Function**: Categorical Crossentropy.
* **Epochs**: Configured for 50 epochs.
* **Callbacks**: 
  * `EarlyStopping`: Monitors validation loss and stops training if it doesn't improve for 5 epochs (restoring the best weights).
  * `ReduceLROnPlateau`: Reduces the learning rate by a factor of 0.2 if the validation loss plateaus for 3 epochs.

---

## 📈 Evaluation & Results
The notebook evaluates the trained model's performance on the validation set using standard machine learning metrics:
* **Classification Report**: Displays precision, recall, and f1-score for each emotion class.
* **Confusion Matrix**: Visualized using a Seaborn heatmap to identify which emotions the model struggles to differentiate.

---

## 🚀 How to Use (Inference)
The project includes a custom `detect_emotion(image_path)` function that allows you to easily pass a new image to the trained model. 

**Steps performed by the function:**
1. Loads the image in grayscale and resizes it to 48x48.
2. Converts the image to an array and scales the pixel values (divide by 255.0).
3. Expands the dimensions to match the model's expected batch input.
4. Outputs the predicted emotion label alongside the model's confidence percentage, and displays the image using Matplotlib.

---

## 📦 Dependencies
The primary libraries required to run this project include:
* `tensorflow` / `keras`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
