'''
ResNet18/34/50/101/152 in TensorFlow2.

Reference:
[1] He, Kaiming, et al. 
    "Deep residual learning for image recognition." 
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
'''
import tensorflow as tf
from tensorflow.keras import layers, models
import sys

class CNN_Encoder(tf.keras.Model):
    expansion = 1
    
    def __init__(self, input_shape, output_activation='softmax', num_classes=10):
        super(CNN_Encoder, self).__init__()
        self._encoder = models.Sequential()
        self._encoder.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        # self._encoder.add(layers.MaxPooling2D((2, 2)))
        self._encoder.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._encoder.add(layers.MaxPooling2D((2, 2)))
        self._encoder.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._encoder.add(layers.Flatten())
        self._encoder.add(layers.Dense(64, activation='relu'))
        self._encoder.add(layers.Dense(num_classes, activation=output_activation))
            
    def call(self, x):
        out = self._encoder(x)
        return out