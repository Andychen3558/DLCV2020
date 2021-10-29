import torch
import torch.nn as nn

class Convnet(nn.Module):
	def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
		super().__init__()
		self.encoder = nn.Sequential(
			conv_block(in_channels, hid_channels),
			conv_block(hid_channels, hid_channels),
			conv_block(hid_channels, hid_channels),
			conv_block(hid_channels, out_channels),
		)

	def forward(self, x):
		x = self.encoder(x)
		return x.view(x.size(0), -1)

def conv_block(in_channels, out_channels):
	bn = nn.BatchNorm2d(out_channels)
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, 3, padding=1),
		bn,
		nn.ReLU(),
		nn.MaxPool2d(2)
	)

class MLP(nn.Module):
	def __init__(self, in_channels=1600, hid_channels=1024, out_channels=512):
		super().__init__()
		self.fc = nn.Sequential(
			nn.Linear(in_channels, hid_channels),
			nn.Dropout(0.3),
			nn.Linear(hid_channels, hid_channels),
			nn.Dropout(0.3),
			nn.Linear(hid_channels, out_channels),
		)

	def forward(self, x):
		return self.fc(x)

class Hallucinator(nn.Module):
	def __init__(self, in_channels=1600, hid_channels=2048, out_channels=1600, noise_dim=1600):
		super().__init__()
		self.fc = nn.Sequential(
			nn.Linear(in_channels+noise_dim, hid_channels),
			nn.ReLU(),
			nn.Linear(hid_channels, out_channels),
		)

	def forward(self, x):
		return self.fc(x)

class Discriminator(nn.Module):
	def __init__(self, in_channels=1600, hid_channels=1024, out_channels=512):
		super(Discriminator, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(in_channels, hid_channels),
			nn.LeakyReLU(0.2),
			nn.Linear(hid_channels, out_channels),
		)
	def forward(self, x):
		return self.fc(x)