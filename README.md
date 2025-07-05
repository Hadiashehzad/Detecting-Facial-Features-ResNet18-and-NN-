# Detecting-Facial-Features-ResNet18-and-NN-
A multi-task deep learning model to detect the gender, presence of glasses, and shirt color from facial images using a ResNet18 backbone with custom classification heads.

Detecting the gender of a person through their image, or detecting whether they’re wearing glasses or not is a useful application in surveillance and security, retail and marketing, and HCI  In this project, I developed a ResNet18-based multi-task deep learning model to detect the gender, presence of glasses, and shirt color from facial images.

# Datasets

## Gender
### FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age
- Source: [GitHub](https://github.com/joojs/fairface)
- Around 98,000 images with labels for age, gender, and race

## Glasses
### People with and without glasses dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/kaiska/apparel-dataset)
- 4,920 labelled images of people’s faces with and without glasses

## Shirts
### Apparel Dataset
- Source: [Kaggle](https://www.kaggle.com/datasets/kaiska/apparel-dataset)
- 16,170 labelled images of people wearing dresses of different categories and different colours
- 
![detecting gender, glasses, shirt](https://github.com/user-attachments/assets/0b1153c7-930a-449e-bd85-ce95e92e4a18)

# Preprocessing

- Ensured equal representation of every race and both genders by filtering out equal number of observations from each category
- Only retained images for shirt, dress, suit, and hoodie out of all apparel categories
- Normalised images according to the mean and sd of images in ImageNet dataset, which was used to pre-train the ResNet18 model.
- Final dataset size:
  - 41,796 observations for gender
  - 4,920 observations for glasses
  - 6.955 observations for shirt

# The Model - ResNet18-based neural network

## Model Overview

I used a pretrained ResNet18 model, a state-of-the-art CNN model designed to classify images, trained on the ImageNet dataset. I utilized all convolutional layers of ResNet18 except for the last fully connected layer which then classifies the images. Then I replaced the last layer with three fully-connected neural networks.

In place of the last fc layer, I added three task-specific heads
- Gender head (binary classifier)
- Glasses head (binary classifier)
- Shirts head (multi-class classifier)
Each head is a small 2-layer MLP (Multi-layer Perceptron) with one hidden layer with activation and one output layer

## Training

I used the Adam optimiser for all models, the binary cross-entropy loss for the gender and glasses heads, and the cross-entropy loss for the shirts head. The final model was trained over 5 epochs . Performance is then measured by tracking the loss over each iteration.

| Head  | Time (using T4 GPU) |
| ------------- | ------------- |
| Gender  | 9m 30 s  |
| Glasses  | 7m 45s  |
| Shirts  | 2m 15s  |

## Results

| Head  | Val Accuracy |  Val Loss  |
| ------------- | ------------- | ------  |
| Gender  | 88.3%  |  37.6%  |
| Glasses  | 100%  |  0.06%  |
| Shirts  | 97.6%  |  8.15%  |

## Testing the model on a new image

![istockphoto-869773424-612x612](https://github.com/user-attachments/assets/e0fc226e-f135-431c-aaa0-9f14ca36ea17)

Prediction: 
- Gender: Male (0.96)
- Glasses: Wearing Glasses (0.63)
- Shirt: Blue Shirt (Class 0)

## Why ResNet18-based NN?

- accuracy and computational efficiency.
- pretrained on ImageNet, allowing powerful feature representations without needing massive training data.
- custom classification heads for each task—predicting gender, detecting glasses, and classifying shirt color
- share learned features across related tasks
- lightweight architecture suitable for real-time or resource-constrained applications.

## Limitations

- lack of a unified dataset containing all three labels—gender, glasses, and shirt color
- the glasses and apparel datasets may still carry inherent biases against gender or race that could affect generalization
- the gender and glasses datasets consisted of cropped facial images, whereas the apparel dataset included full-body shots, leading to a domain mismatch

## Alternatives

- Vision Transformers
- EfficientNet
- ResNet50












