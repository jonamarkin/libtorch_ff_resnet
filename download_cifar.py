import torch
import torchvision
import numpy as np
import os

def download_cifar100_binary():
    # Download using torchvision
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    
    # Create binary directory
    os.makedirs('./data/cifar-100-binary/cifar-100-binary', exist_ok=True)
    
    # Save training data
    with open('./data/cifar-100-binary/cifar-100-binary/train.bin', 'wb') as f:
        for image, fine_label in train_dataset:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            # Write coarse label (0 as placeholder), fine label, and image data
            f.write(bytes([0]))  # coarse label placeholder
            f.write(bytes([fine_label]))  # fine label
            f.write(img_array.tobytes())  # image data
    
    # Save test data
    with open('./data/cifar-100-binary/cifar-100-binary/test.bin', 'wb') as f:
        for image, fine_label in test_dataset:
            img_array = np.array(image)
            f.write(bytes([0]))
            f.write(bytes([fine_label]))
            f.write(img_array.tobytes())

    print("Dataset saved in binary format at ./data/cifar-100-binary/")

if __name__ == "__main__":
    download_cifar100_binary()
