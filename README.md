# lightweight_cnn_mlp

<br>

```bash
lightweight_cnn_mlp/
├── data/                          # CSV + optional image samples
├── models/
│   ├── mlp_tf.py                  # MLP model (Fl.-Only)
│   ├── cnn_tf.py                  # CNN model (Holo.-Only)
│   └── multimodal_tf.py           # Fusion model (Holo.+Fl.)
├── train_mlp.py                   # Training script for Fl.-Only
├── train_cnn.py                   # Training script for Holo.-Only
├── train_multimodal.py            # Training script for Holo.+Fl.
├── evaluate.py                    # Evaluation script for all models
├── requirements.txt               # Updated with tensorflow, pillow etc.
├── README.md                      # Project documentation
└── notebooks/
    └── demo-lightweight_cnn_mlp.ipynb   # Main Colab demo notebook
```


# Lightweight CNN + MLP for MNIST

This project contains a minimal convolutional neural network (CNN) and multi-layer perceptron (MLP) implementation for MNIST digit classification.

## Demo - Google Colab<br>

[demo-lightweight_cnn_mlp](https://colab.research.google.com/drive/1idYEQsNvz0HvnyU9VuenV7-h7eNn6nMP?usp=drive_link)

<br><br>
