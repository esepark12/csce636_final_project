import torch
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
import numpy as np
"""This script implements the functions for data augmentation
and preprocessing.
"""
CIFAR_norm_means = (0.4914, 0.4822, 0.4465)
CIFAR_norm_stds = (0.2470, 0.2435, 0.2616)

def parse_record(record, training):
  """Parse a record to an image and perform data preprocessing.

  Args:
      record: An array of shape [3072,]. One row of the x_* matrix.
      training: A boolean. Determine whether it is in training mode.

  Returns:
      image: An array of shape [32, 32, 3].
  """
  ### YOUR CODE HERE
  image = None
  ### END CODE HERE

  image = preprocess_image(image, training) # If any.

  return image


def preprocess_image(image, training):
  """Preprocess a single image of shape [height, width, depth].

  Args:
      image: An array of shape [32, 32, 3].
      training: A boolean. Determine whether it is in training mode.

  Returns:
      image: An array of shape [32, 32, 3]. The processed image.
  """
  ### YOUR CODE HERE
  if training:
    image = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(CIFAR_norm_means, CIFAR_norm_stds)
    ])
  ### END CODE HERE

  return image


# Other functions
### YOUR CODE HERE

### END CODE HERE