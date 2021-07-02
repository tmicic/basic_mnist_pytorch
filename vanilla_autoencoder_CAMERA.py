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

import cv2 as cv

# cap = the video capture device
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


USE_CUDA = True
ENCODING_SIZE = 128
LR = 0.6
IMG_SIZE = 256

ensure_reproducibility(6000)

device = torch.device("cuda" if USE_CUDA else "cpu")

class Model(nn.Module):
    def __init__(self, encoding_size=128, image_size=64):
        super().__init__()
        self.encoding_size = encoding_size
        self.image_size = image_size
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.encoding = nn.LazyLinear(out_features = encoding_size)
        self.fc = nn.Linear(encoding_size, image_size * image_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        enc = self.encoding(x)
        x = F.relu(enc)
        x = self.fc(x)
        x = x.view(-1,1, self.image_size, self.image_size)
        output = x# torch.sigmoid(x)
        return output, enc, self.conv1.weight

model = Model(encoding_size=ENCODING_SIZE, image_size=IMG_SIZE).to(device=device)
optimizer = optim.Adadelta(model.parameters(), lr=LR)
loss_function = torch.nn.SoftMarginLoss().to(device=device) # torch.nn.BCELoss(reduction='sum').to(device=device)


if __name__ == '__main__':

    
    ret, frame = cap.read()
    last_processed = transforms.F.to_tensor(cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), dsize=(IMG_SIZE, IMG_SIZE))).unsqueeze(0).to(device)
    last_loss = None
    last_encoding = None
    i = 0

    while True:

        i+=1
        ret, frame = cap.read()
        cur_processed = transforms.F.to_tensor(cv.resize(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), dsize=(IMG_SIZE, IMG_SIZE))).unsqueeze(0).to(device)
    
        optimizer.zero_grad()

        output, encoding, monitored_weights = model(last_processed)

        if last_encoding is None:
            last_encoding = encoding.detach()
            encoding_loss = None
        else:
            encoding_loss = F.mse_loss(encoding.detach(), last_encoding)

            last_encoding = encoding.detach()


        loss = loss_function(output, cur_processed)

        if last_loss is None:
            loss.backward()
            optimizer.step()
        else:
            
            loss_change = abs((last_loss-loss.item())/last_loss)
            #if loss_diff >0.01 or i < 100:

            loss.backward()
            optimizer.step()

            print(f'Loss:{loss_change:.4f}, Encoding Loss: {encoding_loss:.4f}', end='\r')            

        last_loss = loss.item()


        last_processed = cur_processed

            #Display the resulting frame
        cv.imshow('frame', output.detach().squeeze().cpu().numpy())



        if cv.waitKey(1) == ord('q'):
            break

