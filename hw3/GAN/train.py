import os
import sys
import random
import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
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

workspace_dir = '.'
# hyperparameters 
batch_size = 64
z_dim = 100
lr = 5e-5
num_epochs = 200
n_critic = 1
clip_value = 0.01
save_dir = os.path.join(workspace_dir, 'logs_wgan')
os.makedirs(save_dir, exist_ok=True)

train_set = ImgDataset(train_x, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(in_dim=z_dim).to(device)
D = Discriminator(3).to(device)
G.train()
D.train()

# loss criterion
criterion = nn.BCELoss()

# optimizer
# opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
# opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)


# for logging
z_sample = Variable(torch.randn(32, z_dim)).to(device)


for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        imgs = data.to(device)
        bs = imgs.size(0)

        """ Train D """
        z = Variable(torch.randn(bs, z_dim)).to(device)
        r_imgs = Variable(imgs).to(device)
        f_imgs = G(z)

        # soft and noisy labels
        # poss = np.random.rand(bs)
        # r_label = torch.from_numpy(np.random.uniform(0.7, 1.2, size=bs)).float().to(device)
        # f_label = torch.from_numpy(np.random.uniform(0.0, 0.3, size=bs)).float().to(device)
        # r_label = torch.ones((bs)).to(device)
        # poss0 = np.random.rand(bs)
        # r_label[poss0 >= 0.8] = 0
        # f_label = torch.zeros((bs)).to(device)
        # poss1 = np.random.rand(bs)
        # f_label[poss1 >= 0.8] = 1

        # dis
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())
        
        # compute loss
        loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
        # r_loss = criterion(r_logit, r_label)
        # f_loss = criterion(f_logit, f_label)
        # loss_D = (r_loss + f_loss) / 2

        # update model
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Clip weights of discriminator
        for p in D.parameters():
            p.data.clamp_(-clip_value, clip_value)

        # Train the generator every n_critic iterations
        if i % n_critic == 0:
            """ train G """
            # leaf
            z = Variable(torch.randn(bs, z_dim)).to(device)
            f_imgs = G(z)

            # dis
            f_logit = D(f_imgs)
            
            # compute loss
            loss_G = -torch.mean(f_logit)
            # loss_G = criterion(f_logit, r_label)

            # update model
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # log
            print(f'\rEpoch [{epoch+1}/{num_epochs}] {i+1}/{len(train_loader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
    
    if (epoch+1) % 5 == 0:
        G.eval()
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join(save_dir, f'Epoch_{epoch+1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=8)
        print(f' | Save some samples to {filename}.')
        

        G.train()
        if (epoch+1) % 5 == 0:
            torch.save(G.state_dict(), os.path.join(workspace_dir, './wgan_g.pth'))
            torch.save(D.state_dict(), os.path.join(workspace_dir, './wgan_d.pth'))

