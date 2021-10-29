import sys
import os
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from hw2_2_model import *
from mean_iou_evaluate import mean_iou_score

def readfile(path, label):
	# label 是一個 boolean variable，代表需不需要回傳 y 值
	image_dir = sorted(os.listdir(path))
	x = []
	masks = []

	for i, file in enumerate(image_dir):
		file_type = file.split('_')[1].split('.')[0] 
		if file_type == 'sat':
			x.append(scipy.misc.imread(os.path.join(path, file)))

		elif label:
			mask = scipy.misc.imread(os.path.join(path, file))
			mask = (mask >= 128).astype(int)
			mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
			gt = np.empty((512, 512))
			gt[mask == 3] = 0  # (Cyan: 011) Urban land 
			gt[mask == 6] = 1  # (Yellow: 110) Agriculture land 
			gt[mask == 5] = 2  # (Purple: 101) Rangeland 
			gt[mask == 2] = 3  # (Green: 010) Forest land 
			gt[mask == 1] = 4  # (Blue: 001) Water 
			gt[mask == 7] = 5  # (White: 111) Barren land 
			gt[mask == 0] = 6  # (Black: 000) Unknown
			gt[mask == 4] = 6  # (Red: 100) Unknown

			masks.append(gt)

	if label:
		return image_dir, np.array(x), np.array(masks)
	else:
		return image_dir, np.array(x)


test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

### Load testing set with readfile function
data_dir = sys.argv[1]
print("Reading data")
# image_dir, test_x, test_y = readfile(data_dir, True)
image_dir, test_x = readfile(data_dir, False)
print("Size of testing data = {}".format(len(test_x)))


class ImgDataset(Dataset):
	def __init__(self, x, y=None, transform=None):
		self.x = x
		self.y = y
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

batch_size = 8
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

### Check if GPU is available, otherwise CPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load model && predict segmentation results
num_class = 7
model = FCN8VGG(num_class).to(device)
model.load_state_dict(torch.load('./improved.pth'))
model.eval()

cls_color = {
	0:  [0, 255, 255],
	1:  [255, 255, 0],
	2:  [255, 0, 255],
	3:  [0, 255, 0],
	4:  [0, 0, 255],
	5:  [255, 255, 255],
	6: [0, 0, 0],
}

masks = []
with torch.no_grad():
	for i, data in enumerate(test_loader):
		data = data.to(device)
		test_pred = model(data)

		masks_batch = torch.max(test_pred, 1)[1].cpu().data.numpy()
		for mask in masks_batch:
			masks.append(mask)

mask_idx = 0
masks = np.array(masks)
# mean_iou_score(masks, test_y)

for i, file in enumerate(image_dir):
	img_idx = int(file.split('_')[0])
	file_type = file.split('_')[1].split('.')[0]

	if file_type == 'sat':
		## draw output segmentation result
		cs = np.unique(masks[mask_idx])
		output = np.zeros((masks[mask_idx].shape[0], masks[mask_idx].shape[1], 3))
		for c in cs:
			ind = np.where(masks[mask_idx]==c)
			output[ind[0], ind[1]] = cls_color[c]
		scipy.misc.imsave(os.path.join(sys.argv[2], '%04d_mask.png' % img_idx), np.uint8(output))

		mask_idx += 1