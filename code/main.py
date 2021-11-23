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

print(torch.__version__)

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
		checkpoint_dir = "ckpt_test.pth"
		checkpoint = torch.load('../saved_models/' + checkpoint_dir)
		model.network.load_state_dict(checkpoint['network'])
		test_acc, correct_samples, total_samples = model.evaluate(test)
		print("Test Result: Total Test Samples %d, Total Correct %d, Test Accuracy %f" %(time.strftime("_%Y-%m-%d_%H%M%S"),total_samples, correct_samples, test_acc))

	elif args.mode == 'predict':
		# Predicting and storing results on private testing dataset
		x_test = load_testing_images(args.data_dir)

		checkpoint_dir = "ckpt_test.pth"
		checkpoint = torch.load('../saved_models/' + checkpoint_dir)
		model.network.load_state_dict(checkpoint['network'])
		predictions = model.predict_prob(x_test)
		save_path = args.save_dir + 'predictions.npy'
		np.save(save_path, predictions)
		data_view = np.load(save_path)
		print("end of predict")


### END CODE HERE

