import utils
import Configure

Conf = Configure.training_configs['learn_rate_schedule']


test = [0,25,175,200,250,275,300,375]

for epoch in test:
    print("Epoch %d LR %f " % (epoch,utils.get_learn_rate(epoch,Conf)))