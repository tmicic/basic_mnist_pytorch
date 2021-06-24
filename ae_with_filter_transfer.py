# adapted from: https://github.com/pytorch/examples/tree/master/mnist
# save and load model: https://pytorch.org/tutorials/beginner/saving_loading_models.html

from numpy import inf
import torch
import torchvision.datasets.mnist as dataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import contextlib
from reproducibility import ensure_reproducibility, seed_worker
import matplotlib.pyplot as plt
import time
import random
import cv2


# TRY:
# MAX POOL AND UN MAX POOL
#
#

import warnings
warnings.filterwarnings("ignore")



TRAIN_BATCH_SIZE = 128
TRAIN_SHUFFLE = False
NUM_WORKERS = 4
PIN_MEMORY = True
USE_CUDA = True
VALIDATE_BATCH_SIZE = 64
VALIDATE_SHUFFLE = False
ENCODING_SIZE = 10

MODEL_PATH = 'mnist.model'
LOAD_AND_CHECK = False

LR = 0.06
EPOCHS = 10

ensure_reproducibility(6000)

device = torch.device("cuda" if USE_CUDA else "cpu")

train_transform = transforms.Compose([transforms.ToTensor()])#,transforms.Normalize((0.1307,), (0.3081,))])
test_transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = dataset.MNIST(download=True, root='data', train=True, transform=train_transform)
validate_dataset = dataset.MNIST(download=True, root='data', train=False, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=TRAIN_SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)
validate_dataloader = DataLoader(validate_dataset, batch_size=VALIDATE_BATCH_SIZE, shuffle=VALIDATE_SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)

class Model(nn.Module):
    def __init__(self, encoding_size=128):
        super().__init__()
        self.encoding_size = encoding_size
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
        self.encoding = nn.Linear(36864, encoding_size)
        self.fc = nn.Linear(encoding_size, 36864)        
        self.upconv1 = nn.ConvTranspose2d(64,32,3,1, bias=False)
        self.upconv2 = nn.ConvTranspose2d(32,1,3,1, bias=False)

        # MAYBE NOT! share weights between conv layers and conv transpose layers. This DRASTICALLY improves the output and coverges MUCH faster
        self.upconv1.weight = self.conv2.weight
        self.upconv2.weight = self.conv1.weight 





    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)        # removing batchnorm - doesnt seem to make a diff = must investigate

        x = F.leaky_relu(x)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = F.leaky_relu(x)
        
        x = torch.flatten(x, 1)


        enc = self.encoding(x)
        x = F.leaky_relu(enc)
        x = self.fc(x)
        x = F.leaky_relu(x)


        x = x.view(-1,64,24,24)
        x = self.upconv1(x)


        x = F.relu(x)
        x = self.upconv2(x)   
        output = torch.sigmoid(x)


        
        return output, enc

model = Model(encoding_size=ENCODING_SIZE).to(device=device)

optimizer = optim.Adadelta(model.parameters())#, lr=LR)       # adadelta only seems to work. cant get adam or SGD to work



# model.fc.requires_grad_(False)
# model.encoding.requires_grad_(False)


loss_function = torch.nn.BCELoss(reduction='sum').to(device=device)
# MSE produces quite eratic behaviour after converging. reduction=mean=very slow to converge.
#
#


if __name__ == '__main__':
    
    batch, _ = next(iter(train_dataloader))


    batch = batch.to(device=device)

    model.train()

    for y in range(10000):
        optimizer.zero_grad()

        output, encoding = model(batch)

        loss = loss_function(output, batch)

        loss.backward()



        optimizer.step()

        if loss.item() < 1600 :
            print(y, "    ", loss.item())
        cv2.imshow('Input', cv2.resize(output.detach().squeeze().cpu()[0].numpy(), (256,256)))

        c = cv2.waitKey(1)
        if c == 27:
            break

