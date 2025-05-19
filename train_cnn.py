import tensorflow as tf
from models.cnn_tf import create_cnn_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define paths - point to your mounted Drive BW dataset folder
train_dir = '/content/drive/MyDrive/Leaf_data/BW'

# Optional: check if data exists
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory not found at: {train_dir}")

# Create image data generator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.4  # 60% train / 40% val split, as per paper
)

# Training generator
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    color_mode='grayscale',  # TIFF BW images
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation generator
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Create CNN model
model = create_cnn_model(input_shape=(128, 128, 1), num_classes=train_generator.num_classes)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save trained model
model.save('/content/lightweight_cnn_mlp/models/cnn_model.h5')
