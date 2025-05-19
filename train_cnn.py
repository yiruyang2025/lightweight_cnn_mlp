import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn_tf import create_cnn_model

# Define absolute dataset paths
train_dir = '/content/lightweight_cnn_mlp/data/holography/train'
val_dir = '/content/lightweight_cnn_mlp/data/holography/val'
model_output_path = '/content/lightweight_cnn_mlp/models/cnn_model.h5'

# Check if data directories exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError(f"Training or validation directory not found:\n{train_dir}\n{val_dir}")

# Create data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical'
)

# Build CNN model
model = create_cnn_model(input_shape=(128, 128, 1), num_classes=train_generator.num_classes)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save model
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
model.save(model_output_path)

print(f"Model successfully trained and saved to: {model_output_path}")
