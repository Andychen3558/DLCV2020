import os
import sys
import numpy as np
import scipy.misc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from model import *

def readfile(path, field):
    ## image
    img_path = os.path.join(path, field)
    image_dir = sorted(os.listdir(img_path))
    x = np.zeros((len(image_dir), 28, 28, 3), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = scipy.misc.imread(os.path.join(img_path, file))
        if len(img.shape) < 3:
            ret = np.empty((28, 28, 3), dtype=np.uint8)
            ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  img
            x[i] = ret
        else:
            x[i] = img

    ## label
    df = pd.read_csv(os.path.join(path, field + '.csv'))
    data = np.array(df)
    fnames = data[:, 0]
    labels = data[:, 1].astype(np.uint8)
    return x, fnames, labels

### Loading images and labels
print("Reading data")
source_dir = sys.argv[1]
src_x, src_fnames, src_y = readfile(source_dir, 'train')
train_src_x, val_src_x, train_src_fnames, val_src_fnames, train_src_y, val_src_y = train_test_split(src_x, src_fnames, src_y, test_size=0.2)
test_src_x, test_src_fnames, test_src_y = readfile(source_dir, 'test')
print("Size of source training data = {}".format(len(train_src_x)))
print("Size of source validation data = {}".format(len(val_src_x)))
print("Size of source testing data = {}".format(len(test_src_x)))
target_dir = sys.argv[2]
tgt_x, tgt_fnames, tgt_y = readfile(target_dir, 'train')
train_tgt_x, val_tgt_x, train_tgt_fnames, val_tgt_fnames, train_tgt_y, val_tgt_y = train_test_split(tgt_x, tgt_fnames, tgt_y, test_size=0.2)
test_tgt_x, test_tgt_fnames, test_tgt_y = readfile(target_dir, 'test')
print("Size of target training data = {}".format(len(train_tgt_x)))
print("Size of target validation data = {}".format(len(val_tgt_x)))
print("Size of target testing data = {}".format(len(test_tgt_x)))


### Preprocessing part
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
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

### Set up hyperparameters
batch_size = 128
lr = 2e-3
num_epochs_pre = 10
num_epochs = 100
beta1, beta2 = 0.5, 0.9
input_dim, hidden_dim, output_dim = 512, 512, 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_encoder = Encoder3().to(device)
tgt_encoder = Encoder3().to(device)
classifier = Classifier().to(device)
critic = Discriminator(input_dim, hidden_dim, output_dim).to(device)


criterion = nn.CrossEntropyLoss()
optimizer_pre = optim.Adam(list(src_encoder.parameters()) + list(classifier.parameters()), lr=lr, betas=(beta1, beta2))


### Dataloader part
train_set = ImgDataset(train_src_x, train_src_y, transform=transform)
train_tgt_set = ImgDataset(train_tgt_x, train_tgt_y, transform=transform)
val_set = ImgDataset(val_tgt_x, val_tgt_y, transform=transform)
val_set2 = ImgDataset(val_src_x, val_src_y, transform=transform)

test_src_set = ImgDataset(test_src_x, test_src_y, transform=transform)
test_set = ImgDataset(test_tgt_x, test_tgt_y, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
train_tgt_loader = DataLoader(train_tgt_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
val_loader2 = DataLoader(val_set2, batch_size=batch_size, shuffle=False, num_workers=0)

test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
test_src_loader = DataLoader(test_src_set, batch_size=batch_size, shuffle=False, num_workers=0)


### Training part

def train_src(train_loader, val_loader):
    best_acc = 0.0

    for epoch in range(num_epochs_pre):
        train_acc ,train_loss = 0.0, 0.0
        train_num = 0
        val_acc ,val_loss = 0.0, 0.0
        val_num = 0

        src_encoder.train()
        classifier.train()
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            optimizer_pre.zero_grad()

            class_logits = classifier(src_encoder(data))
            batch_loss = criterion(class_logits, label)

            batch_loss.backward()
            optimizer_pre.step()

            train_loss += batch_loss.item()
            train_acc += torch.sum(torch.argmax(class_logits, dim=1) == label).item()
            train_num += data.shape[0]

        print('epoch {:>3d}: train loss: {:6.4f}, train acc {:6.4f}'.format(epoch, train_loss / (i+1), train_acc / train_num))
    
        src_encoder.eval()
        classifier.eval()
        with torch.no_grad():
            for i, (data, label) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)

                class_logits = classifier(src_encoder(data))
                batch_loss = criterion(class_logits, label)
                   
                val_loss += batch_loss.item()
                val_acc += torch.sum(torch.argmax(class_logits, dim=1) == label).item()
                val_num += data.shape[0]

        print('epoch {:>3d}: val loss: {:6.4f}, val acc {:6.4f}'.format(epoch, val_loss / i+1, val_acc / val_num))
        acc = val_acc / val_num
        if acc > best_acc:
            best_acc = acc
            print('Best accuracy!')
            torch.save(src_encoder.state_dict(), 'checkpoints/src_encoder2_2.pth')
            torch.save(classifier.state_dict(), 'checkpoints/classifier2_2.pth')

def train_tgt(source_loader, target_loader, val_loader):
    best_acc = 0.0
    src_encoder.load_state_dict(torch.load('checkpoints/src_encoder2_2.pth'))
    tgt_encoder.load_state_dict(torch.load('checkpoints/src_encoder2_2.pth'))
    classifier.load_state_dict(torch.load('checkpoints/classifier2_2.pth'))

    optimizer_tgt = optim.Adam(tgt_encoder.parameters(), lr=0.0001)
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.0002)

    src_encoder.eval()
    classifier.eval()

    for epoch in range(num_epochs):
        train_acc ,train_critic_loss, train_tgt_loss = 0.0, 0.0, 0.0
        val_acc, val_loss, src_acc = 0.0, 0.0, 0.0

        tgt_encoder.train()
        critic.train()
        for i, ((src_data, _), (tgt_data, _)) in enumerate(zip(source_loader, target_loader)):
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)  

            ### train discriminator
            optimizer_critic.zero_grad()

            src_feature = src_encoder(src_data)
            tgt_feature = tgt_encoder(tgt_data)
            src_label = torch.ones(src_feature.size(0)).long()
            tgt_label = torch.zeros(tgt_feature.size(0)).long()
            cat_label = torch.cat((src_label, tgt_label), 0)
            cat_label = cat_label.to(device)

            D_src = critic(src_feature)
            D_tgt = critic(tgt_feature)
            D_out = torch.cat([D_src, D_tgt], 0)
            D_out = D_out.to(device)

            # compute loss for discriminator
            loss_critic = criterion(D_out, cat_label)
            loss_critic.backward()
            optimizer_critic.step()
            train_critic_loss += loss_critic.item()

            # compute accuracy
            pred_cls = torch.squeeze(D_out.max(1)[1])
            acc = (pred_cls == cat_label).float().mean()
            train_acc += acc


            ### train target decoder
            optimizer_tgt.zero_grad()

            tgt_feature = tgt_encoder(tgt_data)
            tgt_pred = critic(tgt_feature)

            # prepare fake labels
            tgt_label = torch.ones(tgt_feature.size(0)).long().to(device)

            # compute loss for target encoder
            loss_tgt = criterion(tgt_pred, tgt_label)
            loss_tgt.backward()
            optimizer_tgt.step()

            train_tgt_loss += loss_tgt

        print('epoch {:>3d}: train d_loss: {:6.4f}, train g_loss: {:6.4f}, train acc {:6.4f}'.format(epoch, train_critic_loss / (i+1), train_tgt_loss / (i+1), train_acc / (i+1)))
    
        tgt_encoder.eval()
        with torch.no_grad():
            for i, (data, label) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)

                class_logits = classifier(tgt_encoder(data))
                batch_loss = criterion(class_logits, label)   
                val_loss += batch_loss.item()

                pred_cls = torch.squeeze(class_logits.max(1)[1])
                acc = (pred_cls == label).float().mean()
                val_acc += acc

        print('epoch {:>3d}: val loss: {:6.4f}, val acc {:6.4f}'.format(epoch, val_loss / (i+1), val_acc / (i+1)))
        Acc = val_acc / (i+1)
        if Acc > best_acc:
            best_acc = Acc
            print('Best accuracy!')
            torch.save(tgt_encoder.state_dict(), './checkpoints/tgt_encoder2_2.pth')


def train():
    ### Train classifier for source domain

    # train_src(train_loader, val_loader2)
    
    train_tgt(train_loader, train_tgt_loader, val_loader)



def visualization(features, labels, domain_labels, fname):  
    X_tsne = TSNE(n_components=2, perplexity=40).fit_transform(features)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    # init color map
    classes = 10
    domains = 2
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    cmap_class = cm.rainbow(np.linspace(0.0, 1.0, classes))
    np.random.shuffle(cmap_class)
    cmap_domain = cm.rainbow(np.linspace(0.0, 1.0, domains))
    np.random.shuffle(cmap_domain)

    for i in range(len(X_norm)):
        ax0.text(X_norm[i, 0], X_norm[i, 1], str(labels[i]), color=cmap_class[labels[i]], fontdict={'weight': 'bold', 'size': 5})
        ax1.text(X_norm[i, 0], X_norm[i, 1], str(domain_labels[i]), color=cmap_domain[domain_labels[i]], fontdict={'weight': 'bold', 'size': 5})
    fig.savefig(fname)

def test():
    tgt_encoder.load_state_dict(torch.load('checkpoints/tgt_encoder3.pth'))
    classifier.load_state_dict(torch.load('checkpoints/classifier3.pth'))
    tgt_encoder.eval()
    classifier.eval()
    Acc = 0

    for i, (data, label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)

        feature = tgt_encoder(data)
        class_logits = classifier(feature)
        batch_loss = criterion(class_logits, label)   

        pred_cls = torch.squeeze(class_logits.max(1)[1])
        acc = (pred_cls == label).float().mean()
        Acc += acc.item()

        # if i == 0:
        #     Features = feature.cpu().detach().numpy().copy()
        # else:
        #     Features = np.concatenate((Features, feature.cpu().detach().numpy()), axis=0)

    # ## extract source domain testing feature
    # for i, (data, label) in enumerate(test_src_loader):
    #     data, label = data.to(device), label.to(device)
    #     feature = tgt_encoder(data)
    #     Features = np.concatenate((Features, feature.cpu().detach().numpy()), axis=0)

    # ## tSNE visualization
    # domains = np.array([1] * test_set.__len__() + [0] * test_src_set.__len__())
    # classes = np.concatenate((test_tgt_y, test_src_y), axis=0)
    # visualization(Features, classes, domains, './figs/StoU.png')

    return Acc / (i+1)

train()
print(test())



