import tensorflow as tf


# Class for the model. In this case, we are using the MobileNetV2 model from Keras
class Model:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = tf.keras.applications.MobileNetV2(
            (32, 32, 3), alpha=0.1, classes=10, weights=None
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        self.model.compile(self.optimizer, self.loss_function, metrics=["accuracy"])

    def get_model(self):
        return self.model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#Pytorch
# class Net1(nn.Module):
#     def __init__(self) -> None:
#         super(Net1, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


#TF
# class Net1(tf.keras.Model):
#     def __init__(self):
#         super(Net1, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(6, (5, 5), activation='relu')
#         self.pool = tf.keras.layers.MaxPooling2D((2, 2))
#         self.conv2 = tf.keras.layers.Conv2D(16, (5, 5), activation='relu')
#         self.flatten = tf.keras.layers.Flatten()
#         self.fc1 = tf.keras.layers.Dense(120, activation='relu')
#         self.fc2 = tf.keras.layers.Dense(84, activation='relu')
#         self.fc3 = tf.keras.layers.Dense(10)

#     def call(self, x):
#         x = self.pool(self.conv1(x))
#         x = self.pool(self.conv2(x))
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

# def get_model(model_name):
#     if model_name == "Net1":
#         return Net1()
#     elif model_name == "resnet18":
#         return models.resnet18(weights=None, num_classes=10)
#     elif model_name == "mobilenet_v2":
#         return models.mobilenet_v2(weights=None, num_classes=10)
#     else:
#         raise ValueError(f"Unknown model name: {model_name}")