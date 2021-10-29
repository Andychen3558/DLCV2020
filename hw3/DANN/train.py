import os
import sys
import numpy as np
import scipy.misc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import time
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
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

### Set up hyperparameters
batch_size = 128
lr = 1e-4
num_epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)
domain_classifier = DomainClassifier().to(device)

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters(), lr=lr)
optimizer_C = optim.Adam(label_predictor.parameters(), lr=lr)
optimizer_D = optim.Adam(domain_classifier.parameters(), lr=lr)


### first part
train_set = ImgDataset(train_src_x, train_src_y, transform=transform)
train_tgt_set = ImgDataset(train_tgt_x, train_tgt_y, transform=transform)
# val_set = ImgDataset(val_src_x, val_src_y, transform=transform)
val_set = ImgDataset(val_tgt_x, val_tgt_y, transform=transform)
test_src_set = ImgDataset(test_src_x, test_src_y, transform=transform)
test_set = ImgDataset(test_tgt_x, test_tgt_y, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
train_tgt_loader = DataLoader(train_tgt_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
test_src_loader = DataLoader(test_src_set, batch_size=batch_size, shuffle=False, num_workers=8)


### Training part
def train_epoch_noDomain(train_loader):
    running_F_loss = 0.0
    total_hit= 0.0

    for i, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        
        feature = feature_extractor(data)
        class_logits = label_predictor(feature)
        
        loss = class_criterion(class_logits, label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == label).item()
        # total_num += source_data.shape[0]
        # print(i, end='\r')

    return running_F_loss / train_set.__len__(), total_hit / train_set.__len__()

def train_epoch(source_dataloader, target_dataloader, lamb):

    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()
        
        # 把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因為在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Label Classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()


        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

def validate_epoch(val_loader):
    label_predictor.eval()
    feature_extractor.eval()
    running_F_loss, total_hit = 0.0, 0.0
    with torch.no_grad():
        for i, (data, label) in enumerate(val_loader):
            data, label = data.to(device), label.to(device)

            feature = feature_extractor(data)
            class_logits = label_predictor(feature)
            
            loss = class_criterion(class_logits, label)
            running_F_loss += loss.item()

            total_hit += torch.sum(torch.argmax(class_logits, dim=1) == label).item()

    return running_F_loss / val_set.__len__(), total_hit / val_set.__len__()


def train():
    best_acc = 0.0
    for epoch in range(num_epochs):
        ### Training models
        train_D_loss, train_F_loss, train_acc = train_epoch(train_loader, train_tgt_loader, lamb=0.1)
        # train_F_loss, train_acc = train_epoch_noDomain(train_loader)

        print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))
        # print('epoch {:>3d}: train F loss: {:6.4f}, train acc {:6.4f}'.format(epoch, train_F_loss, train_acc))

        ### Do validation and update models
        val_F_loss, val_acc = validate_epoch(val_loader)

        # print('epoch {:>3d}: val D loss: {:6.4f}, val F loss: {:6.4f}, val_acc {:6.4f}'.format(epoch, val_D_loss, val_F_loss, val_acc))
        print('epoch {:>3d}: val F loss: {:6.4f}, val acc {:6.4f}'.format(epoch, val_F_loss, val_acc))
        
        if val_acc > best_acc:
            print('[Best accuracy!]')
            best_acc = val_acc
            torch.save(feature_extractor.state_dict(), f'checkpoints2/extractor_model2_dann.pth')
            torch.save(label_predictor.state_dict(), f'checkpoints2/predictor_model2_dann.pth')

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
    feature_extractor.load_state_dict(torch.load('checkpoints2/extractor_model2_dann.pth'))
    label_predictor.load_state_dict(torch.load('checkpoints2/predictor_model2_dann.pth'))
    feature_extractor.eval()
    label_predictor.eval()
    total_hit = 0.0
    for i, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        feature = feature_extractor(data)
        class_logits = label_predictor(feature)
        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == label).item()

        # if i == 0:
        #     Features = feature.cpu().detach().numpy().copy()
        # else:
        #     Features = np.concatenate((Features, feature.cpu().detach().numpy()), axis=0)

    ### extract source domain testing feature
    # for i, (data, label) in enumerate(test_src_loader):
    #     data, label = data.to(device), label.to(device)
    #     feature = feature_extractor(data)
    #     Features = np.concatenate((Features, feature.cpu().detach().numpy()), axis=0)

    ### tSNE visualization
    # domains = np.array([1] * test_set.__len__() + [0] * test_src_set.__len__())
    # classes = np.concatenate((test_tgt_y, test_src_y), axis=0)
    # visualization(Features, classes, domains, './figs/StoU.png')

    return total_hit / test_set.__len__()

train()
print(test())



