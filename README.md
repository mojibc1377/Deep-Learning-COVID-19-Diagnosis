# COVID-19 Detection Using Neural Networks
# this project uses The TPU v2 (Tensor Processing Unit version 2) runtime which is a specialized hardware accelerator developed by Google for machine learning workloads

## Project Description
This project focuses on developing a deep learning model to detect COVID-19 from chest X-ray images. The objective is to create a robust neural network capable of accurately classifying images into COVID-19 positive and negative categories. This project leverages convolutional neural networks (CNNs) for image classification tasks, aiming for high accuracy and reliability.

## Dataset
The dataset consists of chest X-ray images labeled as either COVID-19 positive or negative. Key steps in handling the dataset include:
- **Loading and Preprocessing**: Images are loaded and resized to a uniform shape, normalized to have pixel values between 0 and 1, and split into training and validation sets.
- **Data Augmentation**: Techniques like rotation, zoom, and horizontal flipping are applied to increase the variability of the training data and improve the model's generalization ability.

- <img width="727" alt="Screenshot 2024-06-22 at 16 26 00" src="https://github.com/mojibc1377/Neural-Network-Based-COVID-19-Detection-NN-COVID-Det-/assets/82224660/041e7558-5938-4bc7-9b66-88c2fb589557">

![COVID-1](https://github.com/mojibc1377/Neural-Network-Based-COVID-19-Detection-NN-COVID-Det-/assets/82224660/9ebfdeb3-5821-49f5-9fd0-2c67e367e880)

![COVID-1 2](https://github.com/mojibc1377/Neural-Network-Based-COVID-19-Detection-NN-COVID-Det-/assets/82224660/4e84b2f8-e898-4122-8230-39325f573fb6)


## Code Overview

### Data Preprocessing
- **Loading Data**: Utilize libraries such as TensorFlow and Keras to load images and labels from directories.
- **Resizing and Normalizing**: Convert images to a uniform size (e.g., 224x224 pixels) and normalize pixel values to the [0, 1] range.
- **Splitting Data**: Split the dataset into training and validation sets using an 80-20 split.

### Model Architecture
- **Sequential Model**: A Sequential model from Keras is used to build the CNN.
- **Layers**:
  - **Convolutional Layers**: Extract features from the input images using multiple convolutional layers with ReLU activation.
  - **Pooling Layers**: Downsample the feature maps to reduce dimensionality and computational complexity.
  - **Fully Connected Layers**: Flatten the feature maps and pass them through dense layers for classification.
  - **Output Layer**: A softmax layer to output the probability distribution over the two classes (COVID-19 positive and negative).

### Model Compilation and Training
- **Compilation**: The model is compiled with `categorical_crossentropy` as the loss function, `adam` as the optimizer, and accuracy as the primary metric.
- **Training**: The model is trained for a specified number of epochs (10 in this case) using the training data, with validation at each epoch.
- **Early Stopping and Checkpointing**: Early stopping is employed to halt training if the validation loss does not improve for several epochs, and model checkpoints are saved to keep the best performing model.

### Evaluation
- **Performance Metrics**: The model's accuracy and loss are tracked for both training and validation sets over each epoch.
- **Visualization**: Plot the accuracy and loss to visualize the model's learning progress and identify potential overfitting.

## Training and Evaluation Results

### Graphs
- **Model Accuracy**:
  - Training Accuracy improved consistently, reaching ~91% by the end of epoch 7.
  - Validation Accuracy showed fluctuations, peaking at ~78.45% but indicating potential overfitting.
![modelAcc](https://github.com/mojibc1377/Neural-Network-Based-COVID-19-Detection-NN-COVID-Det-/assets/82224660/4b523cf0-1502-44f6-849e-3084140fc992)


- **Model Loss**:
  - Training Loss decreased steadily, indicating effective learning.
  - Validation Loss was variable, suggesting the need for further regularization or data augmentation.
![modelLoss](https://github.com/mojibc1377/Neural-Network-Based-COVID-19-Detection-NN-COVID-Det-/assets/82224660/8691d667-d9c1-46e1-aea8-5a1dcb68e9f9)


### Detailed Epoch Results
- **Epoch 1**:
  - **Training Loss**: 0.6056
  - **Training Accuracy**: 77.69%
  - **Validation Loss**: 1.3864
  - **Validation Accuracy**: 25.58%
- **Epoch 2**:
  - **Training Loss**: 0.3886
  - **Training Accuracy**: 85.83%
  - **Validation Loss**: 2.7284
  - **Validation Accuracy**: 30.99%
- **Epoch 3**:
  - **Training Loss**: 0.3299
  - **Training Accuracy**: 87.83%
  - **Validation Loss**: 0.7281
  - **Validation Accuracy**: 68.46%
- **Epoch 4**:
  - **Training Loss**: 0.3093
  - **Training Accuracy**: 89.03%
  - **Validation Loss**: 0.5850
  - **Validation Accuracy**: 78.45%
- **Epoch 5**:
  - **Training Loss**: 0.2782
  - **Training Accuracy**: 89.94%
  - **Validation Loss**: 0.9649
  - **Validation Accuracy**: 60.03%
- **Epoch 6**:
  - **Training Loss**: 0.2666
  - **Training Accuracy**: 90.49%
  - **Validation Loss**: 1.8446
  - **Validation Accuracy**: 49.54%
- **Epoch 7**:
  - **Training Loss**: 0.2526
  - **Training Accuracy**: 90.98%
  - **Validation Loss**: 0.6684
  - **Validation Accuracy**: 74.60%

### Analysis
- The training accuracy consistently improved, indicating that the model was learning the training data well.
- The validation accuracy showed variability, which is a sign of overfitting. This could be mitigated by incorporating techniques such as dropout, regularization, or more extensive data augmentation.
- The steady decrease in training loss alongside fluctuating validation loss further supports the potential overfitting issue.

## Conclusion
The CNN model developed for COVID-19 detection from chest X-ray images shows promising training performance but requires further tuning to improve validation accuracy and generalization to new data. Future work could involve experimenting with different model architectures, increasing the dataset size, and applying advanced regularization techniques.

