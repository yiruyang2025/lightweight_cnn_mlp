from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

def build_multimodal(input_shape_img=(224, 224, 3), input_shape_feat=(16,), num_classes=40):
    # CNN branch
    cnn_base = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg', input_shape=input_shape_img)
    cnn_base.trainable = False
    img_input = layers.Input(shape=input_shape_img)
    cnn_features = cnn_base(img_input)

    # MLP branch
    feat_input = layers.Input(shape=input_shape_feat)
    mlp_features = layers.Dense(32, activation='relu')(feat_input)

    # Combine
    combined = layers.Concatenate()([cnn_features, mlp_features])
    x = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=[img_input, feat_input], outputs=output)
