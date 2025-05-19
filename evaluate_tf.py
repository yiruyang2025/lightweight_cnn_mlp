import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from train_tf import MultiModalGenerator, image_size, num_classes

# Load test split and model
test_df = pd.read_csv('data/test_split.csv')
model = tf.keras.models.load_model('model_multimodal.h5')

gen_test = MultiModalGenerator(test_df, batch_size=16, shuffle=False)

# Predict
y_true, y_pred = [], []
for batch_x, batch_y in gen_test:
    preds = model.predict(batch_x)
    y_true.extend(tf.argmax(batch_y, axis=1).numpy())
    y_pred.extend(tf.argmax(preds, axis=1).numpy())

# Scores
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"Weighted F1 Score: {f1:.4f}")
