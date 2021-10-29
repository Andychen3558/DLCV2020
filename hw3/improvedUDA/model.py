import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder1(nn.Module):
	def __init__(self):
		super(Encoder1, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(3, 64, 3, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.Conv2d(64, 128, 3, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.Conv2d(128, 256, 3, 2, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.Conv2d(256, 512, 3, 2, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

			nn.Conv2d(512, 512, 3, 2, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

		)

	def forward(self, x):
		out = self.cnn(x)
		out = out.view(out.size()[0], -1)
		return out

class Encoder2(nn.Module):
	def __init__(self):
		super(Encoder2, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(64, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(128, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(256, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),

		)

	def forward(self, x):
		out = self.cnn(x)
		out = out.view(out.size()[0], -1)
		return out

class Encoder3(nn.Module):
	def __init__(self):
		super(Encoder3, self).__init__()

		self.cnn = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(64, 128, 3, 1, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(128, 256, 3, 1, 1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(256, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(512, 512, 3, 1, 1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)

	def forward(self, x):
		out = self.cnn(x)
		out = out.view(out.size()[0], -1)
		return out


class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(512, 512),
			nn.ReLU(),

			nn.Linear(512, 512),
			nn.ReLU(),

			nn.Linear(512, 10),
		)

	def forward(self, x):
		return self.fc(x)

class Discriminator(nn.Module):
	def __init__(self, input_dims, hidden_dims, output_dims):
		super(Discriminator, self).__init__()

		self.layer = nn.Sequential(
			nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),

            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),

            nn.Linear(hidden_dims, output_dims),
		)

	def forward(self, x):
		return self.layer(x)

