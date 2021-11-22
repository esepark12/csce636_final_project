import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time, sys
import numpy as np
from Network import MyNetwork
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import tqdm
import utils as utils

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.model_configs = configs
        self.network = torch.nn.DataParallel(MyNetwork.ResNetV2Prop())
        #self.network = torch.nn.DataParallel(MyNetwork.ResNetV2Orig(3))
        self.network_swa = AveragedModel(self.network.to('cuda')) # Use SWA Averaged Model
        self.model_setup()

    def model_setup(self):
        pass

    def score(self,outputs,target):
        _,outputs = outputs.max(1) # take the maximum along the 2nd data axis.
        return torch.eq(outputs,target).sum().item() # total correct

    def schedule_lr(self,epoch,config):
        lr = utils.get_learn_rate(epoch,config)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self, train,configs,valid=None,test=None,checkpoint=None):
        
        if valid is not None and test is not None:
            raise('Only supply validation or test data, not both!')
            return

        self.writer = SummaryWriter('../runs/' + configs['experiment_name'] + utils.get_time())
        self.network = self.network.to('cuda')
        
        if checkpoint is not None:
            epoch,accuracy_type,prev_accuracy =  (checkpoint[k] for k in ['epoch','accuracy_type','accuracy'])
            self.network.load_state_dict(checkpoint['net'])
            if accuracy_type == 'test':
                prev_test_accuracy = prev_accuracy
            else:
                prev_valid_accuracy = prev_accuracy
        else:
            epoch = 0
            prev_test_accuracy = 0
            prev_valid_accuracy = 0
    
        #self.network = torch.nn.DataParallel(self.network)
        torch.backends.cudnn.benchmark = True  # good for when input size doesn't change (32x32x3)

        batch_size = configs['batch_size']
        lr = configs['initial_lr']
        self.train_loader = torch.utils.data.DataLoader(train,batch_size,shuffle=True)

        self.optimizer =  torch.optim.SGD(self.network.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
        
        criterion = nn.CrossEntropyLoss()

        max_epoch = configs['max_epoch']
        #scheduler = CosineAnnealingLR(self.optimizer, 100,eta_min=0.0,verbose=True)
        swa_start = 225
        swa_scheduler = SWALR(self.optimizer, swa_lr=0.005)
        
        for epoch in range(max_epoch):
            self.network.train()
            with tqdm.tqdm(total = len(self.train_loader)) as epoch_pbar:
                train_total = 0
                train_correct = 0
                for batch_idx, (x_train, y_train) in enumerate(self.train_loader): # use train loader to enumerate over batches
                    # predict
                    self.optimizer.zero_grad() # set gradients to zero 
                    y_preds = self.network(x_train.to('cuda'))
                    loss = criterion(y_preds, y_train.to('cuda'))

                    # accuracy check
                    train_correct += self.score(y_preds,y_train.to('cuda')) # get running average of train accuracy
                    train_total += len(y_train)
                    train_accuracy = train_correct/train_total

                    # back propagation
                    loss.backward()
                    self.optimizer.step()     

                    # update progress           
                    epoch_pbar.set_description('[training %d/%d] Loss %.2f, Accuracy %.2f' % (epoch,max_epoch,loss, train_accuracy))
                    epoch_pbar.update(1)
        
            self.writer.add_scalar('learning rate',self.optimizer.param_groups[0]['lr'],epoch)
            if epoch > swa_start:
                swa_scheduler.step()
                self.network_swa.update_parameters(self.network)
            else:
                u_lr = self.schedule_lr(epoch,configs['learn_rate_schedule'])
                print("updated lr to %f" % u_lr)

            print('Epoch Done -> Train Total %d, Correct %d' % (train_total,train_correct))
            #scheduler.step(loss) # update learning rate
           
            #self.schedule_lr(epoch,configs['learn_rate_schedule'])
            self.writer.add_scalar('Loss/train',loss,epoch)
            self.writer.add_scalar('Accuracy/train',train_accuracy,epoch)
            

            # check validation accuracy
            if valid is not None:
                valid_accuracy,valid_correct,valid_total = self.evaluate_valid(valid)
                print("Valid Accuracy  %d/%d ---> %.2f | Best ---> %.2f"  % (valid_correct,valid_total,valid_accuracy,prev_valid_accuracy))
                # checkpoint model if accuracy has improved between epochs
                if prev_valid_accuracy  < valid_accuracy:
                    print('[checkpointing model]')
                    utils.checkpoint_model(self.network.state_dict(),self.model_configs['save_dir'],epoch,valid_accuracy,acc_type='valid')
                    prev_valid_accuracy = valid_accuracy

                self.writer.add_scalar('Accuracy/valid',valid_accuracy,epoch)

   
            # check test accuracy
            if test is not None:
                test_accuracy,test_correct,test_total = self.evaluate(test,False)
                #self.writer.add_figure('predictions vs. actual', fig,epoch)
                print("Test Accuracy %d/%d---> %.2f | Best ---> %.2f"  % (test_correct,test_total,test_accuracy,prev_test_accuracy) )
                
                # checkpoint model if accuracy has improved between epochs
                if prev_test_accuracy  < test_accuracy:
                    print('[checkpointing model]')
                    utils.checkpoint_model(self.network.state_dict(),self.model_configs['save_dir'],epoch,test_accuracy)
                    prev_test_accuracy = test_accuracy

                self.writer.add_scalar('Accuracy/test',test_accuracy,epoch)
                if epoch > swa_start:
                    swa_test_accuracy, swa_test_correct, swa_test_total = self.evaluate_swa(test,None)
                    print("SWA Test Accuracy %d/%d---> %.2f"  % (swa_test_correct,swa_test_total,swa_test_accuracy) )
                    self.writer.add_scalar('Accuracy/swa_test',swa_test_accuracy,epoch)
    
            self.writer.flush()
        torch.optim.swa_utils.update_bn(self.train_loader, self.network_swa) # update batch norm parameters in swa

        swa_test_accuracy, swa_test_correct, swa_test_total = self.evaluate_swa(test,None)
        print("[FINAL] SWA Test Accuracy %d/%d---> %.2f"  % (swa_test_correct,swa_test_total,swa_test_accuracy) )
        torch.save(self.network_swa.state_dict(), self.model_configs['save_dir'] + 'ckpt_swa.pth')
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
                    fig = utils.make_image_grid(x_test[random_4],outputs[random_4],y_test[random_4])
                    return test_accuracy,test_correct,test_total, fig
                else:
                    return test_accuracy,test_correct,test_total
                
    def evaluate_swa(self, test,plot_samples_images=False):
        self.network.eval() # set network to evaluation mode
        test_total = 0
        test_correct = 0
        test_loader = torch.utils.data.DataLoader(test,128,shuffle=False)
        with torch.no_grad():
            with tqdm.tqdm(total = len(test_loader)) as test_pbar:
                test_accuracy = 0
                for batch_idx, (x_test, y_test) in enumerate(test_loader):
                    outputs = self.network_swa(x_test.to('cuda'))
                    test_correct += self.score(outputs,y_test.to('cuda'))
                    test_total += len(y_test)
                    test_accuracy = test_correct/test_total
                    test_pbar.set_description('[swa test - Accuracy %.2f]' % (test_accuracy))
                    test_pbar.update(1)
                if plot_samples_images:
                    random_4 = torch.randperm(len(x_test))[:4]
                    fig = utils.make_image_grid(x_test[random_4],outputs[random_4],y_test[random_4])
                    return test_accuracy,test_correct,test_total, fig
                else:
                    return test_accuracy,test_correct,test_total
                
    
    def evaluate_valid(self,valid):
        self.network.eval() # set network to evaluation mode

        valid_loader = torch.utils.data.DataLoader(valid,128,shuffle=False)
        valid_total = 0
        valid_correct = 0
        with torch.no_grad():
            with tqdm.tqdm(total = len(valid_loader)) as valid_pbar:
                valid_accuracy = 0
                for batch_idx, (x_valid, y_valid) in enumerate(valid_loader):
                    outputs = self.network(x_test.to('cuda'))
                    valid_correct += self.score(outputs,y_valid.to('cuda'))
                    valid_total += len(y_valid)
                    valid_accuracy = valid_correct/valid_total
                    valid_pbar.set_description('[valid - Accuracy %.2f]' % (valid_accuracy))
                    valid_pbar.update(1)
        return valid_accuracy,valid_correct,valid_total

    def predict_prob(self, x):
        """
        Function not to be used in Model_SWA

        """
        raise('Use Model.py instead of Model_SWA for network predicitons.')
        pass
### END CODE HERE
