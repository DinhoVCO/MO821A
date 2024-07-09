import tensorflow as tf
from tensorflow.keras import layers, models

# Clase para el modelo LeNet-5
class Model:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(28, 28, 1), padding='same'))
        model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(16, (5, 5), activation='tanh', padding='valid'))
        model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='tanh'))
        model.add(layers.Dense(84, activation='tanh'))
        model.add(layers.Dense(10, activation='softmax'))
        return model

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=["accuracy"])

    def get_model(self):
        return self.model