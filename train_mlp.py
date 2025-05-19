import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from models.mlp_tf import build_mlp

# Load data
csv = pd.read_csv('data/sample_spectra.csv', header=None)
X = csv.iloc[:, 2:18].values
y_raw = csv.iloc[:, 0].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_idx = le.fit_transform(y_raw)
y_encoded = to_categorical(y_idx)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.4, stratify=y_idx, random_state=42)

model = build_mlp((X_train.shape[1],), y_encoded.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2)
model.save('model_mlp_leaf.h5')
