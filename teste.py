import os
import numpy as np
import mahotas.features
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import random_split

# Hyperparameters
batch_size = 32
num_classes = 6
num_epochs = 100
learning_rate = 0.001

class HaralickDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.image_files = []
        self.labels = []
        self.descriptors = []

        # Iterate over subdirectories
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                # Iterate over image files in subdirectory
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if file.endswith('.png'):
                        # Calculate Haralick descriptors for each image
                        image = mahotas.imread(file_path)
                        descriptors = mahotas.features.haralick(image).mean(axis=0)
                        self.image_files.append(file_path)
                        self.labels.append(subdir)
                        self.descriptors.append(descriptors)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = mahotas.imread(self.image_files[idx])
        label = self.labels[idx]
        descriptors = self.descriptors[idx]
        return ToTensor()(image), label, torch.tensor(descriptors)

# Create dataset
dataset = HaralickDataset('sub_images')

# Calculate the sizes of the train and test datasets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)