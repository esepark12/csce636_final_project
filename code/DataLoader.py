import os
import pickle
import numpy as np
from Configure import model_configs, training_configs
import torch, torchvision
import torchvision.transforms as transforms
import PrivateDataset
#import utils
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
    # transform with training augmentation
    if mode is 'train':
        train_transform = training_configs['train_transform']
    else: # test
        train_transform = transforms.Compose([ # defualt transform which only normalizes the data set
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_norm_means, CIFAR_norm_stds) # normalize dataset
        ])

    default_transform = transforms.Compose([ # defualt transform which only normalizes the data set
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_norm_means, CIFAR_norm_stds) # normalize dataset
    ])
    xytrain = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    xytest = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=default_transform)
    xytrain_orig = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=default_transform) #original trainingset without augmentation

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
    default_transform = transforms.Compose([ # defualt transform which only normalizes the data set
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_norm_means, CIFAR_norm_stds) # normalize dataset
    ])
    images = np.load(data_dir + '/private_test_images.npy')
    x_test = PrivateDataset.CSCE_636_PrivateDataset(data_dir,transform=default_transform)
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

