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


# Lightweight CNN + MLP for Leaf Classification

This project implements lightweight convolutional neural network (CNN), multi-layer perceptron (MLP), and multimodal fusion models for leaf species classification using the public **Leaf Dataset** (Silva et al., 2014). 

<br><br>

The dataset contains both:
- **Holographic grayscale images** (from BW directory) for structural patterns,
- **Fluorescence-based numerical features** (from `leaf.csv`) for biochemical descriptors.

<br>

All models are implemented using **TensorFlow** (As in the original paper) and trained/evaluated with accuracy and F1-score metrics.


<br><br>

## Demo - Google Colab<br>

[demo-lightweight_cnn_mlp](https://colab.research.google.com/drive/1idYEQsNvz0HvnyU9VuenV7-h7eNn6nMP?usp=drive_link)

<br><br>
