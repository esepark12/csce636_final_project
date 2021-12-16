import os
import pickle
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from Configure import model_configs, training_configs
import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import data, io
"""This script implements the functions for reading data.
"""
CIFAR_norm_means = (0.4914, 0.4822, 0.4465)
CIFAR_norm_stds = (0.2470, 0.2435, 0.2616)

def load_data(data_dir, mode):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    # training augmentation
    if mode is 'train':
        train_transform = training_configs['train_transform']
    else: # test
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_norm_means, CIFAR_norm_stds)
        ])

    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_norm_means, CIFAR_norm_stds)
    ])
    xytrain = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    xytest = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=default_transform)
    xytrain_orig = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=default_transform) #original train dataset

    ### END CODE HERE

    return xytrain, xytest, xytrain_orig


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_norm_means, CIFAR_norm_stds) # normalize dataset
    ])
    file_path = data_dir + '/private_test_images_v3.npy'
    images = np.load(file_path)
    x_test = PrivateTestImage(file_path,transform=default_transform)
    ### END CODE HERE

    return x_test


def train_valid_split(train, orig_trainset, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """

    ### YOUR CODE HERE
    if train_ratio == 1:
        train_new = train
        return train_new,None

    train_size = int(train_ratio*len(train))
    valid_size = len(train) - train_size
    train_new,_ = torch.utils.data.random_split(train,[train_size,valid_size])
    _, valid = torch.utils.data.random_split(orig_trainset,[train_size,valid_size])

    ### END CODE HERE

    return train_new,valid

class PrivateTestImage(Dataset):

    def __init__(self, file_path, transform=None):
        #self.images = np.load(file_path)

        images = np.load(file_path)
        num_images = images.shape[0]
        self.images = np.empty(shape=(num_images,32,32,3), dtype=np.uint8)
        for i in range(0,num_images):

            self.images[i] = np.reshape(images[i],(32,32,3))

            #For checking the image is in right shape:
            #io.imshow(self.images[i])
            #io.show()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.images[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample