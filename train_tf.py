import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models.multimodal_tf import build_multimodal_model

# Data settings
data_csv = 'data/sample_spectra.csv'
image_dir = 'data/sample_images'
image_size = (224, 224)
num_classes = 10

# Load CSV and split
full_df = pd.read_csv(data_csv)
train_df, test_df = train_test_split(full_df, test_size=0.4, stratify=full_df['label'], random_state=42)
train_df.to_csv('data/train_split.csv', index=False)
test_df.to_csv('data/test_split.csv', index=False)

# Data generator
class MultiModalGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = list(range(len(self.df)))
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.df) / self.batch_size)

    def __getitem__(self, index):
        batch = self.df.iloc[index*self.batch_size:(index+1)*self.batch_size]
        images = []
        spectra = []
        labels = []
        for _, row in batch.iterrows():
            img_path = os.path.join(image_dir, row['filename'])
            img = load_img(img_path, target_size=image_size, color_mode='rgb')
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            spectra.append(row[1:-1].values.astype('float32'))
            labels.append(row[-1])
        return [
            tf.convert_to_tensor(images),
            tf.convert_to_tensor(spectra)
        ], to_categorical(labels, num_classes=num_classes)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

# Build model
model = build_multimodal_model(image_shape=image_size + (3,), spectrum_dim=13, num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare generators
gen_train = MultiModalGenerator(train_df, batch_size=16)
gen_val = MultiModalGenerator(test_df, batch_size=16)

# Train
model.fit(gen_train, validation_data=gen_val, epochs=5)

# Save model
model.save('model_multimodal.h5')
