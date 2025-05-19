from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

def build_efficientnet_cnn(input_shape=(224, 224, 3), num_classes=10):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    base_model.trainable = False  # Freeze the base

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
