import tensorflow as tf
from tensorflow.keras import layers, models

def create_multimodal_model(image_input_shape=(128, 128, 1), vector_input_dim=100, num_classes=10):
    # Image branch
    image_input = layers.Input(shape=image_input_shape)
    x1 = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.Flatten()(x1)

    # Vector branch
    vector_input = layers.Input(shape=(vector_input_dim,))
    x2 = layers.Dense(128, activation='relu')(vector_input)
    x2 = layers.Dense(64, activation='relu')(x2)

    # Concatenate
    combined = layers.concatenate([x1, x2])
    x = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=[image_input, vector_input], outputs=output)
    return model
