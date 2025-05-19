from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

def build_cnn(input_shape=(224, 224, 3), num_classes=40):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg', input_shape=input_shape)
    base_model.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)
