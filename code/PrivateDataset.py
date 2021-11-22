import numpy as np
from torch.utils.data import Dataset
import torch

class CSCE_636_PrivateDataset(Dataset):
    """ Private DataSet for CSCE636 Deep Learning, Fall 2020"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = np.load(root_dir + '/private_test_images.npy')
        self.transform = transform

    def __len__(self):
        return self.images.shape[0] # image batch is shape [n,32,32,3]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.images[idx] # [n,3,32,32]

        if self.transform:
            sample = self.transform(sample)

        return sample