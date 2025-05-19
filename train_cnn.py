import tensorflow as tf
from models.cnn_tf import create_cnn_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Use absolute paths
train_dir = '/content/lightweight_cnn_mlp/data/holography/train'
val_dir = '/content/lightweight_cnn_mlp/data/holography/val'

# Check directories exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise FileNotFoundError(f"Missing train/val directories: {train_dir} or {val_dir}")

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), color_mode='grayscale',
    batch_size=32, class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), color_mode='grayscale',
    batch_size=32, class_mode='categorical')

# Create model
model = create_cnn_model(input_shape=(128, 128, 1), num_classes=train_generator.num_classes)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save model
os.makedirs('/content/lightweight_cnn_mlp/models', exist_ok=True)
model.save('/content/lightweight_cnn_mlp/models/cnn_model.h5')
