# MNIST CNN with PyTorch

A beginner-friendly convolutional neural network project that classifies handwritten digits from the [MNIST dataset](https://yann.lecun.com/exdb/mnist/). The model uses a LeNet-style architecture and is trained with PyTorch.

## What it does

- Downloads and normalizes the MNIST training and test images.
- Splits the 60,000 training images into 90% training and 10% validation data.
- Trains a two-convolution LeNet-style classifier for 10 epochs.
- Saves the best validation-loss checkpoint to `model/lenet_model.pt`.
- Prints final test accuracy and loss.

## Model architecture

| Layer | Configuration |
| --- | --- |
| Input | 1 × 28 × 28 grayscale image |
| Convolution 1 | 6 filters, 5 × 5 kernel, padding 2, ReLU, average pooling |
| Convolution 2 | 16 filters, 5 × 5 kernel, ReLU, average pooling |
| Classifier | Flatten → 120 → 84 → 10 output classes |

## Requirements

- Python 3.9 or later
- PyTorch
- Torchvision

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python MNIST.py
```

The MNIST dataset is downloaded automatically into `data/` on the first run. If a CUDA-capable GPU is available, PyTorch uses it; otherwise the script runs on CPU.

## Project structure

```text
mnist-cnn/
├── MNIST.py             # Data loading, model, training, and evaluation
├── requirements.txt     # Python dependencies
├── .gitignore           # Excludes generated data and checkpoints
└── README.md            # Project documentation
```

## Notes

This is an educational baseline. For reproducible train/validation splits, add a fixed random seed before calling `random_split`.
