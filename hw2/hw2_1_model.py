import numpy as np
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 224, 224]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 2),  # [64, 224, 224]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 2),  # [64, 224, 224]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),    # [64, 112, 112]
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 112, 112]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, 1, 1), # [128, 112, 112]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),    # [128, 56, 56]
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 56, 56]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 56, 56]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 56, 56]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),     # [256, 28, 28]
            nn.Dropout(0.35),
            
            nn.Conv2d(256, 512, 3, 1, 1), # [512, 28, 28]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 28, 28]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 28, 28]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),     # [512, 14, 14]
            nn.Dropout(0.35),

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 14, 14]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 14, 14]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 14, 14]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2, 0),     # [512, 7, 7]
            nn.Dropout(0.35),

        )
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 50)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        cnt = 0
        for layer in self.fc:
            out = layer(out)
            if cnt == 4:
                feature = out
            cnt += 1
        return out, feature
