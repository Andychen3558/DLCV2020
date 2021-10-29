import sys
import os
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import time
from model import *

def readfile(path):
	image_dir = sorted(os.listdir(path))
	x = np.zeros((len(image_dir), 64, 64, 3), dtype=np.uint8)
	for i, file in enumerate(image_dir):
		img = scipy.misc.imread(os.path.join(path, file))
		x[i, :, :] = img
	return x

### Read input face images
data_dir = sys.argv[1]
print("Reading data")
train_x = readfile(os.path.join(data_dir, "train"))
print("Size of training data = {}".format(len(train_x)))


train_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

class ImgDataset(Dataset):
	def __init__(self, x, y=None, transform=None):
		self.x = x
		# label is required to be a LongTensor
		self.y = y
		if y is not None:
			self.y = torch.LongTensor(y)
		self.transform = transform
	def __len__(self):
		return len(self.x)
	def __getitem__(self, index):
		X = self.x[index]
		if self.transform is not None:
			X = self.transform(X)
		if self.y is not None:
			Y = self.y[index]
			return X, Y
		else:
			return X

### Check if GPU is available, otherwise CPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Set up hyperparameters and get dataloader
num_epochs = 200
batch_size = 128
learning_rate = 1e-3
lambda_KL = 1e-4

train_set = ImgDataset(train_x, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

# data = torch.tensor(train_x, dtype=torch.float)
# train_dataset = TensorDataset(data)
# train_sampler = RandomSampler(train_dataset)
# train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

model = VAE().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

Epoch = [i for i in range(1, 1 + num_epochs)]
MSE, KLD = [], []

best_loss = np.inf
model.train()
for epoch in range(num_epochs):
	epoch_start_time = time.time()
	train_loss = 0.0
	mse, kld = 0.0, 0.0

	for i, data in enumerate(train_loader):
		# img = data.float().to(device).permute(0, 3, 1, 2)
		img = data.to(device)
		output = model(img)
		m, k = loss_vae(output[0], img, output[1], output[2], criterion)
		loss = m + lambda_KL * k

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		mse += m.item()
		kld += k.item()

	num_batches = i + 1
	mse, kld = mse/num_batches, kld/num_batches
	avg_loss = mse + lambda_KL * kld
	print('[%03d/%03d] %2.2f sec(s) MSE: %3.6f KLD: %3.6f Loss: %3.6f' % \
		(epoch + 1, num_epochs, time.time()-epoch_start_time, mse, kld, avg_loss))

	if avg_loss < best_loss:
		best_loss = avg_loss
		torch.save(model.state_dict(), sys.argv[2])
		print('Model saved!')

	MSE.append(mse)
	KLD.append(kld)


### Plot learning curves
fig, (mse_curve, kld_curve) = plt.subplots(1, 2, figsize=(10, 3))
fig.subplots_adjust(wspace=0.5)

## MSE curve
mse_curve.set_title('MSE')
mse_curve.plot(Epoch, MSE, color='c')
## KLD curve
kld_curve.set_title('KLD')
kld_curve.plot(Epoch, KLD, color='r')

## Output the curves
fig.savefig('./fig1_2_2.png')