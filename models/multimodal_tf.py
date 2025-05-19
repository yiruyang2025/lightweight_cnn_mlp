from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

def build_multimodal_model(image_shape=(224, 224, 3), spectrum_dim=13, num_classes=10):
    # Image input via EfficientNet
    image_input = layers.Input(shape=image_shape)
    cnn_base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=image_shape, pooling='avg')
    cnn_base.trainable = False
    image_features = cnn_base(image_input)

    # Spectrum input via MLP
    spectrum_input = layers.Input(shape=(spectrum_dim,))
    x_spectrum = layers.Dense(64, activation='relu')(spectrum_input)

    # Combine
    combined = layers.concatenate([image_features, x_spectrum])
    outputs = layers.Dense(num_classes, activation='softmax')(combined)

    return Model(inputs=[image_input, spectrum_input], outputs=outputs)
