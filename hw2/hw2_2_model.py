import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, vgg16_bn

class FCN32VGG(nn.Module):
    def __init__(self, n_classes):
        super(FCN32VGG, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input dimension [3, 512, 512]
        vgg = vgg16(pretrained=True)
        features, classifier =  list(vgg.features.children()), list(vgg.classifier.children())

        features[0].padding = (100, 100)

        for f in features:
            if 'MaxPool2d' in f.__class__.__name__:
                f.ceil_mode = True
            if 'ReLU' in f.__class__.__name__:
                f.inplace = True

        self.features5 = nn.Sequential(*features)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.bn6 = nn.BatchNorm2d(4096)
        self.lrelu6 = nn.LeakyReLU(0.2, inplace=True)
        self.drop6 = nn.Dropout()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.bn7 = nn.BatchNorm2d(4096)
        self.lrelu7 = nn.LeakyReLU(0.2, inplace=True)
        self.drop7 = nn.Dropout()

        self.score_fr = nn.Conv2d(4096, n_classes, 1)
        self.upscore = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=64, stride=32, bias=False)


    def forward(self, x):
        pool5 = self.features5(x)
        h = self.lrelu6(self.bn6(self.fc6(pool5)))
        h = self.drop6(h)

        h = self.lrelu7(self.bn7(self.fc7(h)))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        return h[:, :, 19:(19 + x.size()[2]), 19:(19 + x.size()[3])].contiguous()


class FCN8VGG(nn.Module):
    def __init__(self, n_classes):
        super(FCN8VGG, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input dimension [3, 512, 512]
        vgg = vgg16_bn(pretrained=True)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())

        # features[0].padding = (100, 100)

        for i, f in enumerate(features):
            if 'MaxPool2d' in f.__class__.__name__:
                f.ceil_mode = True
            if 'ReLU' in f.__class__.__name__:
                f.inplace = True
        

        self.features3 = nn.Sequential(*features[:24])
        self.features4 = nn.Sequential(*features[24:34])
        self.features5 = nn.Sequential(*features[34:])

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.bn6 = nn.BatchNorm2d(4096)
        self.lrelu6 = nn.LeakyReLU(0.2, inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.bn7 = nn.BatchNorm2d(4096)
        self.lrelu7 = nn.LeakyReLU(0.2, inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)
        self.score_pool4_0 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn4_0 = nn.BatchNorm2d(1024)
        self.lrelu4_0 = nn.LeakyReLU(0.2, inplace=True)
        self.score_pool4_1 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(2048)
        self.lrelu4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.score_pool4_2 = nn.Conv2d(2048, n_classes, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(n_classes)
        self.lrelu4_2 = nn.LeakyReLU(0.2, inplace=True)

        # self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        # self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=16, stride=8, bias=False)
        # self.upscore_pool4 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=5, stride=3, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=8, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=2, stride=2, bias=False)


    def forward(self, x):
        ## get features after pool3, pool4, pool5
        pool3 = self.features3(x)
        pool4 = self.features4(pool3)
        pool5 = self.features5(pool4)

        h = self.lrelu6(self.bn6(self.fc6(pool5)))
        h = self.drop6(h)

        h = self.lrelu7(self.bn7(self.fc7(h)))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h

        h = self.lrelu4_0(self.bn4_0(self.score_pool4_0(pool4)))
        h = self.lrelu4_1(self.bn4_1(self.score_pool4_1(h)))
        h = self.lrelu4_2(self.bn4_2(self.score_pool4_2(h)))
        # h = h[:, :, 5:5+upscore2.size()[2], 5:5+upscore2.size()[3]]
        score_pool4c = h  #1/16

        h = upscore2 + score_pool4c  #1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  #1/8

        h = self.score_pool3(pool3)
        # h = h[:, :, 9:9+upscore_pool4.size()[2], 9:9+upscore_pool4.size()[3]]
        score_pool3c = h  #1/8

        h = upscore_pool4 + score_pool3c

        h = self.upscore8(h)
        # h = h[:, :, 31:(31 + x.size()[2]), 31:(31 + x.size()[3])].contiguous()
        return h
