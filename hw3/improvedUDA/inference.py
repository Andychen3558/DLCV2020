import os
import sys
import numpy as np
import scipy.misc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model import *


def readfile(path):
    img_path = path
    image_dir = sorted(os.listdir(img_path))
    x = np.zeros((len(image_dir), 28, 28, 3), dtype=np.uint8)
    fnames = []
    for i, file in enumerate(image_dir):
        img = scipy.misc.imread(os.path.join(img_path, file))
        fnames.append(file)
        if len(img.shape) < 3:
            ret = np.empty((28, 28, 3), dtype=np.uint8)
            ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  img
            x[i] = ret
        else:
            x[i] = img

    return x, fnames

### Read argugemts
test_dir = sys.argv[1]
target = sys.argv[2]
output_path = sys.argv[3]

### Loading images
print("Reading data")
test_x, fnames = readfile(test_dir)
print("Size of target testing data = {}".format(len(test_x)))

### Preprocessing part
transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])
transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])
transform3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
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

### Load models and get dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier().to(device)

if target == 'mnistm':
    tgt_encoder = Encoder1().to(device)
    tgt_encoder.load_state_dict(torch.load('improvedUDA/checkpoints/tgt_encoder.pth'))
    classifier.load_state_dict(torch.load('improvedUDA/checkpoints/classifier.pth'))
    transform = transform1
elif target == 'svhn':
    tgt_encoder = Encoder2().to(device)
    tgt_encoder.load_state_dict(torch.load('improvedUDA/checkpoints/tgt_encoder2.pth'))
    classifier.load_state_dict(torch.load('improvedUDA/checkpoints/classifier2.pth'))
    transform = transform1
elif target == 'usps':
    tgt_encoder = Encoder3().to(device)
    tgt_encoder.load_state_dict(torch.load('improvedUDA/checkpoints/tgt_encoder3.pth'))
    classifier.load_state_dict(torch.load('improvedUDA/checkpoints/classifier3.pth'))
    transform = transform1

batch_size = 128
test_set = ImgDataset(test_x, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)



### Testing
tgt_encoder.eval()
classifier.eval()
prediction = []

for i, data in enumerate(test_loader):
    data = data.to(device)

    class_logits = classifier(tgt_encoder(data))
    pred_label = torch.squeeze(class_logits.max(1)[1])
    for y in pred_label:
        prediction.append(y)

### Write prediction results in csv file
with open(output_path, 'w') as f:
    f.write('image_name,label\n')
    assert len(fnames) == len(prediction)
    for i in range(len(fnames)):
        f.write('{},{}\n'.format(fnames[i], prediction[i]))
