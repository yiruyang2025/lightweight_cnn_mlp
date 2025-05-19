from tensorflow.keras import layers, Model

def build_mlp(input_dim=13, num_classes=10):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation='relu')(inputs)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)
