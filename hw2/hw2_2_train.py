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

### 分別將 training set、validation set 用 readfile 函式讀進來
data_dir = sys.argv[1]
print("Reading data")
image_dir, train_x, train_y = readfile(os.path.join(data_dir, "train"), True)
print("Size of training data = {}".format(len(train_x)))
image_dir, val_x, val_y = readfile(os.path.join(data_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def label_to_one_hot(targets: torch.Tensor, n_class):
    """
    get one-hot tensor from targets, ignore the 255 label
    :param targets: long tensor[bs, 1, h, w]
    :param nlabels: int
    :return: float tensor [bs, nlabel, h, w]
    """
    # batch_size, _, h, w = targets.size()
    # res = torch.zeros([batch_size, nlabels, h, w])
    targets = targets.squeeze(dim=1)
    zeros = torch.zeros(targets.shape).long().to(targets.device)

    # del 255.
    targets_ignore = targets > 20
    # print(targets_ignore)
    targets = torch.where(targets <= 20, targets, zeros)

    one_hot = torch.nn.functional.one_hot(targets, num_classes=n_class)
    one_hot[targets_ignore] = 0
    # print(one_hot[targets_ignore])
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


class FocalLoss(nn.Module):

    def __init__(self,
                 mode='CE', n_class=7, mean=True,
                 gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        # self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.mode = mode
        self.n_class = n_class
        self.mean = mean

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        """

        :param input: [bs, c, h, w],
        :param target: [bs, 1, h, w]
        :return: tensor
        """

        if self.mode == 'BCE':
            target = label_to_one_hot(target, n_class=self.n_class)
            pt = input.sigmoid()

            BCE = nn.BCELoss(reduction='none')(pt, target)
            loss = torch.abs(target - pt) ** self.gamma * BCE

        elif self.mode == 'CE':
            if input.dim() > 2:
                input = input.transpose(1, 2).transpose(2, 3).reshape(-1, self.n_class)
                target = target.transpose(1, 2).transpose(2, 3).reshape(-1, 1)
            # print(input.shape, target.shape)

            pt = input.softmax(dim=1)
            pt = pt.gather(dim=1, index=target).view(-1)
            # print(f'pt:{pt.shape}')
            CE = nn.CrossEntropyLoss(reduction='none', ignore_index=255)(input, target.view(-1))
            # print(f'CE:{CE.shape}')
            # print(CE.shape, pt.shape)
            loss = (1 - pt) ** self.gamma * CE
        else:
            raise Exception(f'*** focal loss mode:{self.mode} wrong!')

        if self.mean:
            return loss.mean()
        else:
            return loss.sum()

class ImgDataset(Dataset):
	def __init__(self, x, y=None, transform=None):
		self.x = x
		self.y = y
		# if y is not None:
		# 	self.y = torch.LongTensor(y)
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

### Set up hyper parameters
num_class = 7
num_epoch = 100
best_acc = 0.0
learning_rate = 1e-3

model_type = sys.argv[2]
if model_type == 'fcn8':
	print('training fcn8vgg model!')
	model = FCN8VGG(num_class).to(device)
	train_set = ImgDataset(train_x, train_y, train_transform)
	val_set = ImgDataset(val_x, val_y, test_transform)
else:
	print('training baseline fcn32vgg model!')
	model = FCN32VGG(num_class).to(device)
	train_set = ImgDataset(train_x, train_y)
	val_set = ImgDataset(val_x, val_y)

batch_size = 8
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss = FocalLoss()

# def drawMask(epoch, img_idx, mask, output_path, improved=False):
# 	cls_color = {
# 		0:  [0, 255, 255],
# 		1:  [255, 255, 0],
# 		2:  [255, 0, 255],
# 		3:  [0, 255, 0],
# 		4:  [0, 0, 255],
# 		5:  [255, 255, 255],
# 		6: [0, 0, 0],
# 	}
# 	cs = np.unique(mask)
# 	output = np.zeros((mask.shape[0], mask.shape[1], 3))
# 	for c in cs:
# 		ind = np.where(mask==c)
# 		output[ind[0], ind[1]] = cls_color[c]
# 	if improved:
# 		scipy.misc.imsave(os.path.join(output_path, 'improved_%d_%04d_mask.png' % (epoch, img_idx)), np.uint8(output))
# 	else:
# 		scipy.misc.imsave(os.path.join(output_path, '%d_%04d_mask.png' % (epoch, img_idx)), np.uint8(output))

### Training
for epoch in range(num_epoch):
	epoch_start_time = time.time()
	train_loss = 0.0
	val_loss = 0.0

	if epoch == 70:
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

	model.train()
	pred_train, label_train = np.empty((train_set.__len__(), 512, 512)), np.empty((train_set.__len__(), 512, 512))
	for i, data in enumerate(train_loader):
		data[0], data[1] = data[0].float().to(device), data[1].long().to(device)
		# data[0] = data[0].permute(0, 3, 1, 2)

		optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
		train_pred = model(data[0].to(device)) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
		batch_loss = loss(train_pred, data[1].unsqueeze(1)) # 計算 loss
		# batch_loss = cross_entropy2d(train_pred, data[1], weight=class_weight)

		batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
		optimizer.step() # 以 optimizer 用 gradient 更新參數值

		pred_train[i*batch_size: (i+1)*batch_size] = torch.max(train_pred, 1)[1].cpu().data.numpy()
		label_train[i*batch_size: (i+1)*batch_size] = data[1].cpu().data.numpy()

		train_loss += batch_loss.item()

	print('[%03d/%03d] %2.2f sec(s) Train mIOU: %f Loss: %f' % \
		(epoch + 1, num_epoch, time.time()-epoch_start_time, \
		 mean_iou_score(pred_train, label_train), train_loss/train_set.__len__()))

	model.eval()
	pred_val, label_val = np.empty((val_set.__len__(), 512, 512)), np.empty((val_set.__len__(), 512, 512))
	with torch.no_grad():
		for i, data in enumerate(val_loader):
			data[0], data[1] = data[0].float().to(device), data[1].long().to(device)
			# data[0] = data[0].permute(0, 3, 1, 2)

			val_pred= model(data[0].to(device))
			batch_loss = loss(val_pred, data[1].unsqueeze(1))

			pred_val[i*batch_size: (i+1)*batch_size] = torch.max(val_pred, 1)[1].cpu().data.numpy()
			label_val[i*batch_size: (i+1)*batch_size] = data[1].cpu().data.numpy()
			

			val_loss += batch_loss.item()

		val_acc = mean_iou_score(pred_val, label_val)
		print('[%03d/%03d] %2.2f sec(s) Val mIOU: %f loss: %f' % \
			(epoch + 1, num_epoch, time.time()-epoch_start_time, \
			val_acc, val_loss/val_set.__len__()))
		if val_acc > best_acc:
			best_acc = val_acc
			print('Best accuracy!')
			with open('./log_fcn8.txt', 'w+') as f:
				f.write('best acc = %f' %best_acc)

			if model_type == 'fcn8':
				torch.save(model.state_dict(), './checkpoint/2/improved.pth')
			else:
				torch.save(model.state_dict(), './checkpoint/2/baseline.pth')

		## draw segmentation mask during training
		# indices = [10, 97, 107]
		# if epoch+1 == 1 or epoch+1 == 20 or epoch+1 == 40:
		# 	print('[drawing segmentation masks for report!]')
		# 	for idx in indices:
		# 		drawMask(epoch+1, idx, pred_val[idx], './mask_10_97_107', improved=True)