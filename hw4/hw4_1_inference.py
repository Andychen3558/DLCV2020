import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd
import scipy.misc
from utils import euclidean_dist, cosine_similarity
from model import Convnet, MLP

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# trainable_array = nn.Linear(512, 512).to(device)
# trainable_array.load_state_dict(torch.load('./checkpoint/p1_parametric.pth'))

def readfile(path, csv_path):
    df = pd.read_csv(csv_path)
    data = np.array(df)

    ## image
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 84, 84, 3), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = scipy.misc.imread(os.path.join(path, file))
        x[i] = img

    ## label
    labels = data[:, 2]
    return x, labels


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        X = self.x[index]
        X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        return X

    def __len__(self):
        return len(self.x)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path, n_classes):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)


def predict(args, encoder, mlp, data_loader):

    prediction_results = []
    encoder.eval()
    mlp.eval()

    with torch.no_grad():

        ## each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):

            data = data.to(device)
            rst = []

            ## create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])

            ## extract the feature of support and query data
            z = mlp(encoder(data))
            z_dim = z.size(-1)

            z_proto = z[:args.N_way * args.N_shot].view(args.N_way, args.N_shot, z_dim).mean(1)
            z_query = z[args.N_way * args.N_shot:]

            ## calculate the distance between prototype & query features
            dists = euclidean_dist(z_query, z_proto)
            # dists = cosine_similarity(z_query, z_proto)
            # dists = torch.mm(trainable_array(z_query), z_proto.transpose(0, 1))
            score = -dists

            ## classify the query data depending on the its distense with each prototype
            pred_label = torch.squeeze(score.max(1)[1])
            for y in pred_label:
                rst.append(y.item())
            prediction_results.append(rst)

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    ### Load images and labels
    print("Reading data")
    test_x, test_y = readfile(args.test_data_dir, args.test_csv)
    print("Size of testing data = {}".format(len(test_x)))

    ### load the models
    test_classes = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Convnet().to(device)
    mlp = MLP().to(device)
    encoder.load_state_dict(torch.load('./checkpoints/p1_encoder_1.pth'))
    mlp.load_state_dict(torch.load('./checkpoints/p1_mlp_1.pth'))


    ### Get datasets and dataloaders
    test_dataset = MiniDataset(test_x, test_y)
    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv, test_classes))

    prediction_results = predict(args, encoder, mlp, test_loader)

    ### output the prediction to csv
    with open(args.output_csv, 'w') as f:
        f.write('episode_id')
        for i in range(args.N_way * args.N_query):
            f.write(',query' + str(i))
        f.write('\n')
        for i in range(len(prediction_results)):
            f.write(str(i) + ',')
            f.write(','.join(str(x) for x in prediction_results[i]))
            f.write('\n')