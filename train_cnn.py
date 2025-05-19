from models.cnn_tf import build_cnn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

BATCH_SIZE = 16
IMG_SIZE = (224, 224)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(
    'data/sample_images',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

val_gen = datagen.flow_from_directory(
    'data/sample_images',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

model = build_cnn(input_shape=(224, 224, 3), num_classes=train_gen.num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=20)
model.save('model_cnn_leaf.h5')
