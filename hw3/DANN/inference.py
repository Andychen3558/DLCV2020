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

### Load models and get dataloader
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = FeatureExtractor().to(device)
label_predictor = LabelPredictor().to(device)

if target == 'mnistm':
    feature_extractor.load_state_dict(torch.load('DANN/checkpoints2/extractor_model0_dann.pth'))
    label_predictor.load_state_dict(torch.load('DANN/checkpoints2/predictor_model0_dann.pth'))
elif target == 'svhn':
    feature_extractor.load_state_dict(torch.load('DANN/checkpoints2/extractor_model1_dann.pth'))
    label_predictor.load_state_dict(torch.load('DANN/checkpoints2/predictor_model1_dann.pth'))
elif target == 'usps':
    feature_extractor.load_state_dict(torch.load('DANN/checkpoints2/extractor_model2_dann.pth'))
    label_predictor.load_state_dict(torch.load('DANN/checkpoints2/predictor_model2_dann.pth'))


test_set = ImgDataset(test_x, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)



### Testing
feature_extractor.eval()
label_predictor.eval()
prediction = []

for i, data in enumerate(test_loader):
    data = data.to(device)

    feature = feature_extractor(data)
    class_logits = label_predictor(feature)
    pred_label = torch.squeeze(class_logits.max(1)[1])
    for y in pred_label:
        prediction.append(y)

### Write prediction results in csv file
with open(output_path, 'w') as f:
    f.write('image_name,label\n')
    assert len(fnames) == len(prediction)
    for i in range(len(fnames)):
        f.write('{},{}\n'.format(fnames[i], prediction[i]))