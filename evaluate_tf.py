
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_cnn(model_path, image_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        image_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    return y_true, y_pred

def evaluate_mlp(model_path, csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns='label').values
    y = df['label'].values
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X)
    y_pred = np.argmax(predictions, axis=1)
    return y, y_pred

def evaluate_multimodal(model_path, image_dir, csv_path):
    from models.multimodal_tf import load_multimodal_data  # You need to implement this function
    X_img, X_csv, y_true = load_multimodal_data(image_dir, csv_path)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict([X_img, X_csv])
    y_pred = np.argmax(predictions, axis=1)
    return y_true, y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['cnn', 'mlp', 'multimodal'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--csv_path', type=str, default=None)
    args = parser.parse_args()

    if args.model_type == 'cnn':
        y_true, y_pred = evaluate_cnn(args.model_path, args.image_dir)
    elif args.model_type == 'mlp':
        y_true, y_pred = evaluate_mlp(args.model_path, args.csv_path)
    elif args.model_type == 'multimodal':
        y_true, y_pred = evaluate_multimodal(args.model_path, args.image_dir, args.csv_path)

    print(classification_report(y_true, y_pred))
