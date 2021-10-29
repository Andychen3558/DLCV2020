import sys
import os
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# from sklearn.manifold import TSNE
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
from hw2_1_model import *

def readfile(path, label):
	# label 是一個 boolean variable，代表需不需要回傳 y 值
	image_dir = sorted(os.listdir(path))
	x = np.zeros((len(image_dir), 32, 32, 3), dtype=np.uint8)
	y = np.zeros((len(image_dir)), dtype=np.uint8)
	for i, file in enumerate(image_dir):
		file_names.append(file)
		img = scipy.misc.imread(os.path.join(path, file))
		img = img[..., ::-1]
		x[i, :, :] = img
		if label:
			y[i] = int(file.split('_')[0])
	if label:
		return x, y
	else:
		return x

### Load testing set with readfile function
test_dir = sys.argv[1]
print("Reading data")
file_names = []
test_x = readfile(test_dir, False)
# test_x, test_y = readfile(test_dir, True)
print("Size of testing data = {}".format(len(test_x)))

test_transform = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((224, 224)),						   
	transforms.ToTensor(),
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

batch_size = 32
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

### Check if GPU is available, otherwise CPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load model && predict
model = Classifier().to(device)
model.load_state_dict(torch.load('./model_p1.pth'))
model.eval()

prediction = []
acc = 0
with torch.no_grad():
	for i, data in enumerate(test_loader):
		test_pred, feature_secLast = model(data.to(device))
		test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
		for y in test_label:
			prediction.append(y)

		# acc += np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == test_y[i*batch_size:(i+1)*batch_size])

	# 	feature_secLast = feature_secLast.cpu().numpy()
	# 	if i == 0:
	# 		Features = feature_secLast.copy()
	# 	else:
	# 		Features = np.concatenate((Features, feature_secLast), axis=0)
	
	# print('Acc = {:.6f}'.format(acc/test_set.__len__()))
	
### Visualization
def visualization(features, labels, fname):
	X_tsne = TSNE(n_components=2, perplexity=50).fit_transform(features)
	x_min, x_max = X_tsne.min(0), X_tsne.max(0)
	X_norm = (X_tsne - x_min) / (x_max - x_min)
	# init color map
	classes = 50
	fig, ax = plt.subplots(figsize=(10, 10))
	# cmap = plt.get_cmap('gist_rainbow', classes)
	cmap = cm.rainbow(np.linspace(0.0, 1.0, classes))
	np.random.shuffle(cmap)
	last = -1
	for i in range(len(X_norm)):
		plt.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=cmap[labels[i]], fontdict={'weight': 'bold', 'size': 10})
	plt.savefig(fname)


# visualization(Features, test_y, './visualization.png')

### Write prediction results in csv file
with open(os.path.join(sys.argv[2], 'test_pred.csv'), 'w') as f:
	f.write('image_id,label\n')
	assert len(file_names) == len(prediction)
	for i in range(len(file_names)):
		f.write('{},{}\n'.format(file_names[i], prediction[i]))
