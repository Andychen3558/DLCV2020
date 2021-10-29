import sys
import os
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from hw2_1_model import *


def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 32, 32, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = scipy.misc.imread(os.path.join(path, file))
        img = img[..., ::-1]
        x[i, :, :] = img
        if label:
            y[i] = int(file.split('_')[0])
    if label:
        return x, y
    else:
        return x

### 分別將 training set、validation set 用 readfile 函式讀進來
data_dir = sys.argv[1]
print("Reading data")
train_x, train_y = readfile(os.path.join(data_dir, "train_50"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(data_dir, "val_50"), True)
print("Size of validation data = {}".format(len(val_x)))


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomRotation(15), #隨機旋轉圖片
    transforms.RandomPerspective(),
    transforms.RandomAffine(15),
    transforms.ColorJitter(brightness=(0.5, 1.5),contrast=(0.5, 1.5),saturation=(0.5, 1.5)),
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
])

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
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

### Check if GPU is available, otherwise CPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Set up hyper parameters
num_epoch = 150
best_acc = 0.0
learning_rate = 0.001

model = Classifier().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### Training
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc ,train_loss = 0.0, 0.0
    val_acc ,val_loss = 0.0, 0.0

    if epoch == 100:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate*0.1, momentum=0.9)

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred, _ = model(data[0].to(device)) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device)) # 計算 loss
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
        (epoch + 1, num_epoch, time.time()-epoch_start_time, \
         train_acc/train_set.__len__(), train_loss/train_set.__len__()))


    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred, _ = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))
               
            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()


        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
        acc = val_acc/val_set.__len__()
        if acc > best_acc:
            best_acc = acc
            print('Best accuracy!')
            torch.save(model.state_dict(), './checkpoint/model_1.pth')
