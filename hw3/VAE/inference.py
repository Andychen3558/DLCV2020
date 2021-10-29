import sys
import os
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
from model import *

# def readfile(path):
# 	# label 是一個 boolean variable，代表需不需要回傳 y 值
# 	image_dir = sorted(os.listdir(path))
# 	x = np.zeros((len(image_dir), 64, 64, 3), dtype=np.uint8)
# 	for i, file in enumerate(image_dir):
# 		img = scipy.misc.imread(os.path.join(path, file))
# 		x[i, :, :] = img
# 	return x

### Read input face images
# data_dir = './hw3_data/face'
# print("Reading data")
# test_x = readfile(os.path.join(data_dir, "test"))
# print("Size of testing data = {}".format(len(test_x)))


# test_transform = transforms.Compose([
# 	transforms.ToPILImage(),
# 	transforms.ToTensor(),
# 	transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
# ])

# class ImgDataset(Dataset):
# 	def __init__(self, x, y=None, transform=None):
# 		self.x = x
# 		# label is required to be a LongTensor
# 		self.y = y
# 		if y is not None:
# 			self.y = torch.LongTensor(y)
# 		self.transform = transform
# 	def __len__(self):
# 		return len(self.x)
# 	def __getitem__(self, index):
# 		X = self.x[index]
# 		if self.transform is not None:
# 			X = self.transform(X)
# 		if self.y is not None:
# 			Y = self.y[index]
# 			return X, Y
# 		else:
# 			return X

### Check if GPU is available, otherwise CPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load model
model = VAE().to(device)
model.load_state_dict(torch.load('./VAE/vae.pth'))
model.eval()

# batch_size = 128
# test_set = ImgDataset(test_x, transform=test_transform)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

### problem 1-3 --- reconstruction
# output_path = './output1-3'
# criterion = nn.MSELoss()
# num_choosed = 10
# choosed = np.random.choice(len(test_x), num_choosed)
# with torch.no_grad():
# 	for idx in choosed:
# 		img = test_set.__getitem__(idx).unsqueeze(0).to(device)
# 		output = model(img)

# 		x, recon_x = img.squeeze(0).permute(1, 2, 0).cpu().numpy(), output[0].squeeze(0).permute(1, 2, 0).cpu().numpy()
# 		x, recon_x = (x * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255, (recon_x * [0.5, 0.5, 0.5] + [0.5, 0.5, 0.5]) * 255
		
# 		scipy.misc.imsave(os.path.join(output_path, '%05d.png' % (idx + 40000)), np.uint8(x))
# 		scipy.misc.imsave(os.path.join(output_path, '%05d_recon.png' % (idx + 40000)), np.uint8(recon_x))
# 		mse = criterion(output[0], img)
# 		print('Image index: %d MSE: %3.6f' %(idx+40000, mse))

### problem 1-5 --- tSNE
# isBlondHair = []
# with open(os.path.join(data_dir, 'test.csv'), 'r') as f:
# 	attrs = f.readline()
# 	for line in f.readlines():
# 		x = line.split(',')[4]
# 		isBlondHair.append(int(float(x)))

# with torch.no_grad():
# 	for i, data in enumerate(test_loader):
# 		img = data.to(device)
# 		z = model.encoder(img)
# 		z = torch.flatten(z, start_dim=1).cpu().numpy()

# 		# mu, logvar = model.encode(img)
# 		# z = model.reparametrize(mu, logvar).cpu().numpy()
# 		if i == 0:
# 			Features = z.copy()
# 		else:
# 			Features = np.concatenate((Features, z), axis=0)

# def visualization(features, labels, fname):
# 	X_tsne = TSNE(n_components=2, perplexity=40).fit_transform(features)
# 	x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# 	X_norm = (X_tsne - x_min) / (x_max - x_min)
# 	fig = plt.figure(figsize=(10, 8))
# 	ax = fig.add_subplot(111)
# 	# init color map
# 	num_classes = 2
# 	cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
# 	np.random.shuffle(cmap)

# 	table = {0: 'Not blond hair', 1: 'Blond hair'}
# 	labeled = [False] * num_classes
# 	for i in range(len(X_norm)):
# 		if labeled[labels[i]] is False:
# 			ax.scatter(X_norm[i][0], X_norm[i][1], s=10, marker='o', color=cmap[labels[i]], label=table[labels[i]])
# 		else:
# 			ax.scatter(X_norm[i][0], X_norm[i][1], s=10, marker='o', color=cmap[labels[i]])
# 		labeled[labels[i]] = True
# 	box = ax.get_position()
# 	ax.set_position([box.x0, box.y0, box.width*0.87, box.height])
# 	ax.legend(bbox_to_anchor=(1.02, 0.2))
# 	plt.savefig(fname)

# visualization(Features, isBlondHair, './fig1_5_2.png')


### Generate images
num_samples = 32
imgs_sample = model.sample(num_samples, device, fixed_seed=True)
imgs_sample = (imgs_sample.data + 1) / 2.0
filename = sys.argv[1]
torchvision.utils.save_image(imgs_sample, filename, nrow=8)