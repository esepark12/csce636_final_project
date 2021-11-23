import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time, sys
import numpy as np
from Network import MyNetwork
#from Network2 import MyNetwork
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class MyModel(object):

    def __init__(self, configs):
        self.model_configs = configs
        self.network = torch.nn.DataParallel(MyNetwork.getMyMobileNet())
        self.model_setup()

    def model_setup(self):
        pass

    def score(self,outputs,target):
        _,outputs = outputs.max(1) # take the maximum along the 2nd data axis.
        return torch.eq(outputs,target).sum().item() # total correct

    def schedule_lr(self,epoch,config):
        keys = config.keys()
        for key in keys:
            if epoch < key:
                lr = config[key]
        lr = config[key]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return

    def train(self, train,configs,valid=None,test=None,checkpoint=None):

        self.network = self.network.to('cuda')

        if valid is not None:
            if test is not None:
                raise('Cannot do validation and test at the same time')
            return

        self.writer = SummaryWriter('../runs/' + configs['experiment_name'] + time.strftime("_%Y-%m-%d_%H%M%S"))

        if checkpoint is not None:
            epoch,accuracy_type,prev_accuracy =  (checkpoint[k] for k in ['epoch','accuracy_type','accuracy'])
            self.network.load_state_dict(checkpoint['net'])
            if accuracy_type == 'test':
                prev_test_accuracy = prev_accuracy
            else:
                prev_valid_accuracy = prev_accuracy
        else:
            prev_test_accuracy = 0
            prev_valid_accuracy = 0
            epoch = 0

        torch.backends.cudnn.benchmark = True

        batch_size = configs['batch_size']
        lr = configs['initial_lr']
        self.train_loader = torch.utils.data.DataLoader(train,batch_size,shuffle=True)

        self.optimizer =  torch.optim.SGD(self.network.parameters(), lr=lr,
                                          momentum=0.9, weight_decay=5e-4)

        scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        criterion = nn.CrossEntropyLoss()

        max_epoch = configs['max_epoch']
        checkpoint_filename = "ckpt_test.pth"
        for epoch in range(max_epoch):
            self.network.train()
            with tqdm.tqdm(total = len(self.train_loader)) as epoch_pbar:
                train_total = 0
                train_correct = 0
                for batch_idx, (x_train, y_train) in enumerate(self.train_loader):
                    # predict
                    self.optimizer.zero_grad() # set gradients to zero
                    y_preds = self.network(x_train.to('cuda'))
                    loss = criterion(y_preds, y_train.to('cuda'))

                    # get accuracy
                    train_correct += self.score(y_preds,y_train.to('cuda'))
                    train_total += len(y_train)
                    train_accuracy = train_correct/train_total

                    # back-propagation
                    loss.backward()
                    self.optimizer.step()

                    # update progress
                    epoch_pbar.set_description('[training %d/%d] Loss %.2f, Accuracy %.2f' % (epoch,max_epoch,loss, train_accuracy))
                    epoch_pbar.update(1)

            print('Epoch Done -> Train Total %d, Correct %d' % (train_total,train_correct))
            self.schedule_lr(epoch,configs['learn_rate_schedule'])
            self.writer.add_scalar('Loss/train',loss,epoch)
            self.writer.add_scalar('Accuracy/train',train_accuracy,epoch)
            self.writer.add_scalar('learning rate',self.optimizer.param_groups[0]['lr'],epoch)

            # check test accuracy
            if test is not None:
                test_accuracy,test_correct,test_total,fig = self.evaluate(test,True)
                self.writer.add_figure('predictions vs. actual', fig,epoch)
                print("Test Accuracy %d/%d---> %.2f | Best ---> %.2f"  % (test_correct,test_total,test_accuracy,prev_test_accuracy) )

                # update checkpoint model if accuracy improved than last epoche
                if prev_test_accuracy  < test_accuracy:
                    print('[checkpointing model]')
                    dir = self.model_configs['save_dir']
                    if not os.path.isdir(dir):
                        os.mkdir(dir)
                    state = {
                        'network': self.network.state_dict(),
                        'epoch': epoch,
                        'accuracy': test_accuracy,
                        'accuracy_type': 'test'
                    }
                    torch.save(state, dir + checkpoint_filename)
                    prev_test_accuracy = test_accuracy

                self.writer.add_scalar('Accuracy/test',test_accuracy,epoch)


            self.writer.flush()
        self.writer.close()
        return

    def evaluate(self, test,plot_samples_images=False):
        self.network.eval() # set network to evaluation mode
        test_total = 0
        test_correct = 0
        test_loader = torch.utils.data.DataLoader(test,128,shuffle=False)
        with torch.no_grad():
            with tqdm.tqdm(total = len(test_loader)) as test_pbar:
                test_accuracy = 0
                for batch_idx, (x_test, y_test) in enumerate(test_loader):
                    outputs = self.network(x_test.to('cuda'))
                    test_correct += self.score(outputs,y_test.to('cuda'))
                    test_total += len(y_test)
                    test_accuracy = test_correct/test_total
                    test_pbar.set_description('[test - Accuracy %.2f]' % (test_accuracy))
                    test_pbar.update(1)
                if plot_samples_images:
                    random_4 = torch.randperm(len(x_test))[:4]

                    images = x_test[random_4]
                    outputs_rand = outputs[random_4]
                    labels = y_test[random_4]
                    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                                         std = [ 1/0.2470, 1/0.2435, 1/0.2616 ]),
                                                    transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
                                                                         std = [ 1., 1., 1. ]),
                                                    ])
                    # detach variables from cuda
                    images = [invTrans(images[i].detach()).cpu().numpy() for i in range(len(images))]
                    _, preds = outputs_rand.max(1)
                    preds = preds.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()

                    preds_tmp = np.squeeze(preds)
                    probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds_tmp, outputs_rand.float())] #TODO - fix probabilities.

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

                    return test_accuracy,test_correct,test_total, fig
                else:
                    return test_accuracy,test_correct,test_total

    def predict_prob(self, x):
        self.network.eval()

        private_loader = torch.utils.data.DataLoader(x,100,shuffle=False)
        outputs = torch.empty(len(x),100) #10
        with torch.no_grad():
            with tqdm.tqdm(total = len(private_loader)) as private_pbar:
                for batch_idx, xi in enumerate(private_loader):
                    output = self.network(xi.float().cuda())
                    outputs[batch_idx*len(output):(batch_idx+1)*len(output)] = output
                    private_pbar.update(1)

        probs = F.softmax(outputs,dim=1).to('cpu').numpy()
        num_rows, num_cols = probs.shape
        new_probs = probs[:num_rows,:10]
        return new_probs
### END CODE HERE