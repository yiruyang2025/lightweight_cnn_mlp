# lightweight_cnn_mlp

<br>

```bash
lightweight_cnn_mlp/
├── data/
│   ├── sample_images/           ← Small image subset for testing
│   └── sample_spectra.csv       ← Spectra features + labels
├── models/
│   ├── cnn.py
│   ├── mlp.py
│   └── multimodal.py            ← CNN + MLP fusion model
├── train.py                     ← Full training loop
├── evaluate.py                  ← Evaluation script with F1
├── requirements.txt             ← Dependencies
├── README.md
└── notebooks/
    └── demo-lightweight_cnn_mlp.ipynb ← Main demo notebook for Colab
```


# Lightweight CNN + MLP for MNIST

This project contains a minimal convolutional neural network (CNN) and multi-layer perceptron (MLP) implementation for MNIST digit classification.

## Demo - Google Colab


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yiruyang2025/lightweight_cnn_mlp/blob/main/notebooks/main.ipynb)

<br><br>
