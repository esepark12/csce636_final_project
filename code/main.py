### YOUR CODE HERE
# import tensorflow as tf
import torch
import time
import os, argparse
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs


parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
parser.add_argument("data_dir", help="path to the data")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
	model = MyModel(model_configs)

	if args.mode == 'train':
		train,test,orig_trainset = load_data(args.data_dir, args.mode)
		train,valid = train_valid_split(train,orig_trainset,train_ratio=1)

		model.train(train, training_configs,valid=None,test=test,checkpoint=None)
		model.evaluate(test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		train,test,orig_trainset = load_data(args.data_dir, args.mode)
		checkpoint_dir = "ckpt.pth"#"best_ckpt_ResNetProp_Standard.pth"
		checkpoint = torch.load('../saved_models/' + checkpoint_dir)
		model.network.load_state_dict(checkpoint['net'])
		test_accuracy, correct, total = model.evaluate(test)
		print("[%s%s test results] Model Accuracy %f, Total Correct %d, Total Test Samples %d" %(checkpoint_dir,time.strftime("_%Y-%m-%d_%H%M%S"),test_accuracy,correct,total))

	elif args.mode == 'predict':
		# Predicting and storing results on private testing dataset
		x_test = load_testing_images(args.data_dir)

		#checkpoint_dir = "best_ckpt_ResNetProp_Standard.pth"
		checkpoint_dir = "ckpt.pth"
		checkpoint = torch.load('../saved_models/' + checkpoint_dir)
		model.network.load_state_dict(checkpoint['net'])
		predictions = model.predict_prob(x_test)
		np.save(args.save_dir, predictions)
		data_view = np.load(args.save_dir + '.npy')
		print("end of predict")


### END CODE HERE

