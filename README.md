# Hand Gesture Recognition using Leap Motion Dataset

## Problem Statement
The objective of this project is to develop a hand gesture recognition system using the Leap Motion dataset. The system will accurately identify and classify various hand gestures performed by different subjects based on near-infrared images captured by the Leap Motion sensor. The goal is to facilitate intuitive human-computer interaction and gesture-based control systems in real-world applications.

## Project Overview
This project leverages the Leap Motion dataset, which includes 10 different hand gestures performed by 10 subjects (5 men and 5 women). Each gesture is represented by a series of infrared images. The project involves the following key steps:

- **Data Loading and Preprocessing:** Extracting and preparing the dataset for training and evaluation.
- **Model Development:** Implementing a convolutional neural network (CNN) architecture for gesture recognition.
- **Training and Evaluation:** Training the model on the dataset and evaluating its performance using validation and test sets.
- **Inference:** Deploying the trained model to predict hand gestures from unseen images.

## Dataset
The dataset consists of infrared images captured by the Leap Motion sensor. It includes 10 different hand gestures performed by 10 subjects. The dataset is organized in a structured manner where images are grouped into folders based on the gesture type and subject.

## Data Preprocessing
1. **Loading Images:** Images are read using OpenCV and resized to (64, 64) pixels for uniformity.
2. **Normalization:** Images are normalized by scaling pixel values between 0 and 1.
3. **Label Encoding:** Gesture labels are converted into categorical format.
4. **Train-Test Split:** The dataset is split into training, validation, and test sets using an 80-10-10 split.

## Model Architecture
The model is implemented using TensorFlow and consists of:
- **Convolutional Layers:** Extract features from the images.
- **Max-Pooling Layers:** Reduce spatial dimensions.
- **Fully Connected Layers:** Classify gestures.
- **Dropout Layers:** Prevent overfitting.

### Model Summary
- Input Shape: (64, 64, 3)
- Convolutional Layers: 3
- Max-Pooling Layers: 3
- Fully Connected Layers: 2
- Activation Function: ReLU
- Output Layer: Softmax (11 classes)
- Optimizer: Adam
- Loss Function: Categorical Crossentropy

## Model Training
- **Epochs:** 20
- **Batch Size:** 32
- **Training Accuracy:** Achieved high accuracy on training and validation sets.
- **Loss Reduction:** Significant improvement observed over epochs.

## Results
The model demonstrated exceptional accuracy in recognizing hand gestures. The validation accuracy reached nearly 100%, indicating the model's effectiveness in classifying gestures.

## Conclusion
The project successfully implemented a CNN-based hand gesture recognition system using the Leap Motion dataset. The model demonstrated high accuracy, making it suitable for real-world applications such as virtual reality, gaming, and assistive technologies.

## Future Enhancements
- Integrating real-time gesture recognition.
- Expanding dataset diversity for improved generalization.
- Deploying the model using a web or mobile interface.

## Requirements
To run this project, install the following dependencies:
```
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

## How to Run
1. Clone the repository.
2. Place the Leap Motion dataset in the appropriate folder.
3. Run the Python script to preprocess data and train the model.
4. Evaluate the model using the test dataset.

## Acknowledgment
Special thanks to the Leap Motion dataset creators and Prodigy Infotech for providing the internship opportunity to work on this project.

