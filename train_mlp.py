import pandas as pd
import tensorflow as tf
from models.mlp_tf import create_mlp_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load data
data = pd.read_csv('data/fluorescence/data.csv')

# Preprocess data
X = data.drop('label', axis=1).values
y = LabelEncoder().fit_transform(data['label'])
y = to_categorical(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Create model
model = create_mlp_model(input_dim=X_train.shape[1], num_classes=y.shape[1])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save model
model.save('models/mlp_model.h5')
