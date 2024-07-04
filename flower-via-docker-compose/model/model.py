import tensorflow as tf
from tensorflow.keras import layers, models
   
#MNIST
class Net2:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
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

#CIFAR-10    
# class Net2:
#     def __init__(self, learning_rate):
#         self.learning_rate = learning_rate
#         self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
#         self.model = models.Sequential(
#             [
#                 layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation="relu"),
#                 layers.MaxPooling2D(pool_size=(2, 2)),
#                 layers.Conv2D(64, (3, 3), activation="relu"),
#                 layers.MaxPooling2D(pool_size=(2, 2)),
#                 layers.Flatten(),
#                 layers.Dropout(0.5),
#                 layers.Dense(10, activation="softmax")
#             ]
#         )   
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

#     def compile(self):
#         self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])

#     def get_model(self):
#         return self.model

