import tensorflow as tf
from tensorflow.keras import layers, models

def create_mlp_model(input_dim=100, num_classes=10):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
