# AlexNet Implementation on CIFAR-10
This project implements AlexNet architecture on the CIFAR-10 dataset using PyTorch. The implementation is modularized for better code organization and reusability.

## Project Structure
```
.
├── data_loader.py      # Dataset and data loader utilities
├── model.py            # AlexNet model architecture
├── train.py            # Training and evaluation code
├── data/               # Directory for dataset storage
├── checkpoints/        # Directory for saved model weights
└── README.md
```

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy

You can install the requirements using:

```bash
pip install -r requirements.txt
```

## Instructions to Run the Project

1. Make sure all requirements are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

The script will:
- Download the CIFAR-10 dataset automatically if not already present
- Train the AlexNet model for the specified number of epochs
- Display training progress and validation accuracy
- Save the best model in the checkpoints directory

## Model Architecture
The implementation follows the AlexNet architecture with modifications to handle CIFAR-10 images:
- 5 convolutional layers
- 3 fully connected layers
- Batch normalization after each convolutional layer
- ReLU activation functions
- Dropout in fully connected layers
