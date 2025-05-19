import pandas as pd
import tensorflow as tf
from models.multimodal_tf import create_multimodal_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# Load vector data
vector_data = pd.read_csv('data/fluorescence/data.csv')
X_vector = vector_data.drop('label', axis=1).values
y = LabelEncoder().fit_transform(vector_data['label'])
y = to_categorical(y)

# Load image data
image_dir = 'data/holography/images'
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

# Ensure the number of images matches the number of vector samples
assert len(image_files) == len(X_vector), "Mismatch between image and vector data samples."

# Load and preprocess images
X_image = []
for file in image_files:
    img = tf.keras.preprocessing.image.load_img(file, color_mode='grayscale', target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    X_image.append(img_array)
X_image = np.array(X_image)

# Split data
X_image_train, X_image_val, X_vector_train, X_vector_val, y_train, y_val = train_test_split(
    X_image, X_vector, y, test_size=0.2, random_state=42)

# Scale vector data
scaler = StandardScaler()
X_vector_train = scaler.fit_transform(X_vector_train)
X_vector_val = scaler.transform(X_vector_val)

# Create model
model = create_multimodal_model(image_input_shape=(128, 128, 1), vector_input_dim=X_vector.shape[1], num_classes=y.shape[1])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit([X_image_train, X_vector_train], y_train, epochs=10, validation_data=([X
::contentReference[oaicite:40]{index=40}
