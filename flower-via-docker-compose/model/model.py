import tensorflow as tf
from tensorflow.keras import layers, models

# Class for the model. In this case, we are using the MobileNetV2 model from Keras
# class Model:
#     def __init__(self, learning_rate, dataset):
#         self.learning_rate = learning_rate
#         self.dataset = dataset
#         self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
#         if self.dataset == "mnist":
#             size = (28, 28, 1)
#         else:
#             size = (32, 32, 3)
#         self.model = tf.keras.applications.MobileNetV2(
#             size, alpha=0.9, classes=10, weights=None
#         )
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

#     def compile(self):
#         self.model.compile(self.optimizer, self.loss_function, metrics=["accuracy"])

#     def get_model(self):
#         return self.model
    
# Net1
# class Net1:
#     def __init__(self, learning_rate):
#         self.learning_rate = learning_rate
#         self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
#         self.model = models.Sequential(
#             [
#                 layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
#                 layers.MaxPooling2D(pool_size=(2, 2)),
#                 layers.Conv2D(64, (5, 5), activation="relu"),
#                 layers.MaxPooling2D(pool_size=(3, 3)),
#                 layers.Conv2D(64, (3, 3), activation="relu"),
#                 layers.Flatten(),
#                 layers.Dense(64, activation="relu"),
#                 layers.Dense(10)
#             ]
#         )   
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

#     def compile(self):
#         self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])

#     def get_model(self):
#         return self.model
    

class Net2:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                #layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="softmax")
            ]
        )   
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])

    def get_model(self):
        return self.model

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models

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


# def get_model(model_name):
#     if model_name == "Net1":
#         return Net1()
#     elif model_name == "resnet18":
#         return models.resnet18(weights=None, num_classes=10)
#     elif model_name == "mobilenet_v2":
#         return models.mobilenet_v2(weights=None, num_classes=10)
#     else:
#         raise ValueError(f"Unknown model name: {model_name}")