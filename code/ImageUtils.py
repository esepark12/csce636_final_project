import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
import imgaug as ia
import utils 

"""This script implements the functions for data augmentation
and preprocessing.
"""

def compute_channel_mean_std(tensor):
    '''
    Computes the per channel mean / standard deviation used for
    normalization in the image transformations.
    '''
    mean = torch.mean(tensor,[0,2,3]) # assumes tensor is
    return


class ImgAugTransformStandard:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Fliplr(0.5), # random flip (horizontal)
        iaa.size.CropAndPad(keep_size=True), # random crops
    ])

  def __call__(self, img):
    img = np.array(img)
    
    return self.aug.augment_image(img) 

class ImgAugTransform1:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.size.CropAndPad(keep_size=True), # random crops
        iaa.Fliplr(0.5), # random flips
        iaa.Affine(translate_percent={'x':(-0.1,0.1), 'y':(-0.1,0.1)}), # random translations
        iaa.Affine(rotate=(-20, 20), mode='symmetric'), # random rotations
    ])

  def __call__(self, img):
    img = np.array(img)
    
    return self.aug.augment_image(img)


class ImgAugTransform2:
  
  def __init__(self):
      self.aug = iaa.Sequential([
        iaa.size.CropAndPad(keep_size=True), # random crops
        iaa.Fliplr(0.5), # random flips
        iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-20, 20),
        shear=(-8, 8)
    )
    ])
    
  def __call__(self, img):
    img = np.array(img)
    
    return self.aug.augment_image(img)


ImgTransform1 = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(10),
  transforms.RandomAffine(0,translate=(0.1,0.1)),
  transforms.RandomGrayscale(p=0.2),
  transforms.ToTensor(),
  transforms.Normalize(utils.cifar_10_norm_mean, utils.cifar_10_norm_std)
])

ImgTransform2 = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(20),
  transforms.RandomAffine(0,translate=(0.2,0.2)),
  transforms.RandomGrayscale(p=0.2),
  transforms.ToTensor(),
  transforms.Normalize(utils.cifar_10_norm_mean, utils.cifar_10_norm_std)
])

ImgTransform3 = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.RandomGrayscale(p=0.1),
  transforms.ToTensor(),
  transforms.Normalize(utils.cifar_10_norm_mean, utils.cifar_10_norm_std)
])

ImgTransformStandard = transforms.Compose([
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(utils.cifar_10_norm_mean, utils.cifar_10_norm_std)
])
