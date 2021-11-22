import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg') # command-line use only
import matplotlib.pyplot as plt
import os
import time

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

cifar_10_norm_mean = (0.4914, 0.4822, 0.4465)
cifar_10_norm_std = (0.2470, 0.2435, 0.2616)

def get_learn_rate(epoch,config):
    keys = config.keys()
    for key in keys:
        if epoch < key:
            return config[key]
    return config[key] # return last learning rate if epoch exceeds 

def make_image_grid(images,outputs,labels):
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.2470, 1/0.2435, 1/0.2616 ]),
                                    transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                        std = [ 1., 1., 1. ]),
                                ])
    # detach variables from cuda
    images = [invTrans(images[i].detach()).cpu().numpy() for i in range(len(images))]
    _, preds = outputs.max(1)
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    probs = preds_to_probs(preds,outputs)
    img_grid = plot_classes_preds(images,preds,probs,labels)
    return img_grid

def preds_to_probs(preds,output):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    # convert output probabilities to predicted class
    preds = np.squeeze(preds)
    return [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output.float())] #TODO - fix probabilities.


def plot_classes_preds(images,preds,probs,labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    # plot the images in the batch, along with predicted and true labels
    plt.ioff()
    fig = plt.figure(figsize=(12, 3))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        image = images[idx]
        image[image < 0] = 0.0 # do this to supress matplotlib clipping warning
        plt.imshow(np.transpose(image.astype(np.float), (1, 2, 0)),interpolation='nearest', aspect='equal')#matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig

def checkpoint_model(network_state,checkpoint_dir,epoch,acc,acc_type='test'):
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    state = {
        'net': network_state,
        'accuracy': acc,
        'epoch': epoch,
        'accuracy_type' : acc_type
    }
    torch.save(state, checkpoint_dir + 'ckpt.pth')

def get_time():

    return time.strftime("_%Y-%m-%d_%H%M%S")
