import sys
import os
import random
import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from model import *

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

### Check if GPU is available, otherwise CPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load pretrained model
z_dim = 100
G = Generator(z_dim).to(device)
# G.load_state_dict(torch.load('./drop_g.pth'))
G.load_state_dict(torch.load('./GAN/wgan_g.pth'))
G.eval()

### Generate images
same_seeds(0)
num_samples = 32
z_sample = Variable(torch.randn(num_samples, z_dim)).to(device)
imgs_sample = (G(z_sample).data + 1) / 2.0
filename = sys.argv[1]
torchvision.utils.save_image(imgs_sample, filename, nrow=8)