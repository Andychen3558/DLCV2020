import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
import scipy.misc
from utils import euclidean_dist, cosine_similarity
from model import Convnet, MLP


def readfile(path, csv_path):
	df = pd.read_csv(csv_path)
	data = np.array(df)

	## image
	image_dir = sorted(os.listdir(path))
	x = np.zeros((len(image_dir), 84, 84, 3), dtype=np.uint8)
	for i, file in enumerate(image_dir):
		img = scipy.misc.imread(os.path.join(path, file))
		x[i] = img

	## label
	labels = data[:, 2]
	return x, labels

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):														  
	np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
	def __init__(self, x, y=None):
		self.x = x
		self.y = y

		self.transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

	def __getitem__(self, index):
		X = self.x[index]
		X = self.transform(X)
		if self.y is not None:
			Y = self.y[index]
			return X, Y
		return X

	def __len__(self):
		return len(self.x)

class GeneratorSampler(Sampler):
	def __init__(self, episode_file_path, n_classes):
		if episode_file_path == None:
			## For training
			self.sampled_sequence = []
			for e in range(args.N_episode):
				img_indices, query_indices = [], []
				cls_chosen = np.random.choice(n_classes, args.N_way_train, replace=False)
				for c in cls_chosen:
					img_chosen = np.random.choice(list(range(c*args.N_total_shots, (c+1)*args.N_total_shots)), args.N_shot+args.N_query, replace=False)
					img_indices.extend(img_chosen[:args.N_shot])
					query_indices.extend(img_chosen[args.N_shot:])
				img_indices.extend(query_indices)
				self.sampled_sequence.extend(img_indices)
		else:
			## For validation/testing
			episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
			self.sampled_sequence = episode_df.values.flatten().tolist()
		

	def __iter__(self):
		return iter(self.sampled_sequence) 

	def __len__(self):
		return len(self.sampled_sequence)


def parse_args():
	parser = argparse.ArgumentParser(description="Few shot learning")
	parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
	parser.add_argument('--N-way-train', default=10, type=int, help='N_way_train (default: 10)')
	parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
	parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
	parser.add_argument('--N-episode', default=600, type=int, help='N_episode (default: 600)')
	parser.add_argument('--N-total-shots', default=600, type=int, help='N_total_shots (default: 600)')
	parser.add_argument('--load', type=str, help="Model checkpoint path")
	parser.add_argument('--train_data_dir', type=str, help="Training images directory")
	parser.add_argument('--train_csv', type=str, help="Training images csv file")
	parser.add_argument('--val_data_dir', type=str, help="Validation images directory")
	parser.add_argument('--val_csv', type=str, help="Validation images csv file")
	parser.add_argument('--val_testcase_csv', type=str, help="Validation test case csv")

	return parser.parse_args()

if __name__=='__main__':
	args = parse_args()

	### Load images and labels
	print("Reading data")
	train_x, train_y = readfile(args.train_data_dir, args.train_csv)
	val_x, val_y = readfile(args.val_data_dir, args.val_csv)
	print("Size of training data = {}".format(len(train_x)))
	print("Size of validation data = {}".format(len(val_x)))

	### Set up hyperparameters and load models
	train_classes = 64
	val_classes = 16
	val_shot = 1
	batch_size0 = args.N_way_train * (args.N_query + args.N_shot)
	batch_size1 = args.N_way * (args.N_query + val_shot)
	lr = 1e-3
	# num_epochs = 150
	num_epochs = 50
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	encoder = Convnet().to(device)
	mlp = MLP().to(device)
	# trainable_array = nn.Linear(512, 512).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(list(encoder.parameters()) + list(mlp.parameters()), lr=lr)
	# optimizer = optim.SGD(list(encoder.parameters()) + list(mlp.parameters()) + list(trainable_array.parameters()), lr=lr)


	### Get datasets and dataloaders
	train_dataset = MiniDataset(train_x, train_y)
	val_dataset   = MiniDataset(val_x, val_y)
	

	val_loader = DataLoader(
		val_dataset, batch_size=batch_size1,
		num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
		sampler=GeneratorSampler(args.val_testcase_csv, val_classes))

	best_acc = 0.0

	### Training and validation
	for epoch in range(num_epochs):
		epoch_start_time = time.time()
		train_acc ,train_loss = 0.0, 0.0
		train_num = 0
		val_acc ,val_loss = 0.0, 0.0
		val_num = 0

		train_loader = DataLoader(
			train_dataset, batch_size=batch_size0,
			num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
			sampler=GeneratorSampler(None, train_classes))


		encoder.train()
		mlp.train()
		# trainable_array.train()

		## each batch represent one episode (support data + query data)
		for i, (data, target) in enumerate(train_loader):

			data = data.to(device)

			## create the relative label (0 ~ N_way-1) for query data
			label_encoder = {target[n * args.N_shot] : n for n in range(args.N_way_train)}
			query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way_train * args.N_shot:]])

			## extract the feature of support and query data
			z = mlp(encoder(data))
			z_dim = z.size(-1)

			z_proto = z[:args.N_way_train * args.N_shot].view(args.N_way_train, args.N_shot, z_dim).mean(1)
			z_query = z[args.N_way_train * args.N_shot:]

			## calculate the distance between prototype & query features
			dists = euclidean_dist(z_query, z_proto)
			# dists = cosine_similarity(z_query, z_proto)
			# dists = torch.mm(trainable_array(z_query), z_proto.transpose(0, 1))
			score = -dists

			## calculate loss and update model parameters
			loss = criterion(score, query_label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_acc += torch.sum(torch.argmax(score, dim=1) == query_label).item()
			train_num += dists.shape[0]
			train_loss += loss.item()
			
		print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
			(epoch + 1, num_epochs, time.time()-epoch_start_time, \
			train_acc/train_num, train_loss/(i+1)))


		encoder.eval()
		mlp.eval()
		# trainable_array.eval()

		with torch.no_grad():
			for i, (data, target) in enumerate(val_loader):

				data = data.to(device)

				## create the relative label (0 ~ N_way-1) for query data
				label_encoder = {target[n * val_shot] : n for n in range(args.N_way)}
				query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * val_shot:]])

				## extract the feature of support and query data
				z = mlp(encoder(data))
				z_dim = z.size(-1)

				z_proto = z[:args.N_way * val_shot].view(args.N_way, val_shot, z_dim).mean(1)
				z_query = z[args.N_way * val_shot:]

				## calculate the distance between prototype & query features
				dists = euclidean_dist(z_query, z_proto)
				# dists = cosine_similarity(z_query, z_proto)
				# dists = torch.mm(trainable_array(z_query), z_proto.transpose(0, 1))
				score = -dists

				## calculate loss
				loss = criterion(score, query_label)

				val_acc += torch.sum(torch.argmax(score, dim=1) == query_label).item()
				val_num += dists.shape[0]
				val_loss += loss.item()
				
			print('[%03d/%03d] %2.2f sec(s) Val Acc: %3.6f Loss: %3.6f' % \
				(epoch + 1, num_epochs, time.time()-epoch_start_time, \
				val_acc/val_num, val_loss/(i+1)))

			if val_acc/val_num > best_acc:
				best_acc = val_acc/val_num
				print('Best accuracy!')
				torch.save(encoder.state_dict(), './checkpoints/p1_encoder_3-2.pth')
				torch.save(mlp.state_dict(), './checkpoints/p1_mlp_3-2.pth')
				# torch.save(trainable_array.state_dict(), 'checkpoint/p1_parametric.pth')