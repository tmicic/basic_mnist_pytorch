from typing import ForwardRef
from numpy import inf
import torch
from torch.nn import parameter
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
from tqdm import tqdm     # https://www.youtube.com/watch?v=RKHopFfbPao&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=50
from utils import ignore_warnings
from time import sleep 

# Set plot window to maximised: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python/18824814#18824814
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())


TRAIN_BATCH_SIZE = 32
TRAIN_SHUFFLE = False
NUM_WORKERS = 4
PIN_MEMORY = True
USE_CUDA = True
VALIDATE_BATCH_SIZE = 512
VALIDATE_SHUFFLE = False
ENCODING_SIZE = 128
USE_SCALER = True
ENSURE_REPRODUCIBILITY = True


MODEL_PATH = 'mnist.model'
LOAD_AND_CHECK = False

LR = 0.06
EPOCHS = 10

if ENSURE_REPRODUCIBILITY: ensure_reproducibility(6000, debug_only=False)

device = torch.device("cuda" if USE_CUDA else "cpu")

train_transform = transforms.Compose([transforms.ToTensor()])#,transforms.Normalize((0.1307,), (0.3081,))])
test_transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = dataset.MNIST(download=True, root='data', train=True, transform=train_transform)
validate_dataset = dataset.MNIST(download=True, root='data', train=False, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=TRAIN_SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)
validate_dataloader = DataLoader(validate_dataset, batch_size=VALIDATE_BATCH_SIZE, shuffle=VALIDATE_SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)

class SelfLearnerModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.cortex = nn.Conv2d(32, 1, 1, 1)

        self.fc = nn.Linear(5 * 5, 28 * 28)

    def get_cortex_representation(self, X):
        X = self.conv1(X)
        X = F.leaky_relu(X)
        X = F.max_pool2d(X, 2)
        X = self.conv2(X)
        X = F.leaky_relu(X)
        X = F.max_pool2d(X, 2)
        X = self.cortex(X)

        return X

    def forward(self, X):

        cortex = self.get_cortex_representation(X)
        X = torch.flatten(F.relu(cortex), 1)
        X = self.fc(X)
        X = X.view(-1,1,28,28)
        X = torch.sigmoid(X)
        

        return X, cortex

model = SelfLearnerModel().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=LR)
loss_function = torch.nn.BCELoss(reduction='sum').to(device=device)

if MODEL_PATH != '' and LOAD_AND_CHECK:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

if __name__ == '__main__':

    best_loss = inf
    
    for epoch in range(0, EPOCHS + 1):

        if not LOAD_AND_CHECK and epoch == 0:   # ignore epoch 0 if we are not loading and checking a previously saved model.
            continue
    
        if epoch == 0:
            print(f'Validating previously saved model {MODEL_PATH}...')
        else:
            print(f'Epoch: {epoch} of {EPOCHS}')

        for is_training, dataloader, context in [(True, train_dataloader, contextlib.nullcontext()), (False, validate_dataloader, torch.no_grad())]:

            total_loss = 0.

            if epoch == 0 and is_training:
                continue

            if is_training:
                model.train()
            else:
                model.eval()
            
            with tqdm(total=len(dataloader)) as pbar:
                if is_training:
                    pbar.set_description_str('Training')
                else:
                    pbar.set_description_str('Validating')

                with context:
                    
                    
                    for i, (X, y) in enumerate(dataloader):
                        
                        X = X.to(device)
                        y = y.to(device)

                        if is_training: optimizer.zero_grad()

                        output, cortex = model(X)

                        loss = loss_function(output, X)

                        if is_training:
                            loss.backward()
                            optimizer.step()

                        total_loss += loss.item()

                        pbar.set_postfix_str(f'Loss: {total_loss:.2f}')
                        pbar.update()

                    if not is_training:
                        
                        if len(X) > 10:
                            img = torch.cat([X[:10].view(-1, 28), output[:10].view(-1, 28)], dim=1).detach().cpu()
                        else:
                            img = torch.cat([X.view(-1, 28), output.view(-1, 28)], dim=1).detach().cpu()
                        
                        plt.imshow(img)
                        plt.show(block=False)
                        plt.pause(0.001)

    exit()



    # for i in range(10):
    #     sleep(0.5)
        
    #     pbar.set_postfix_str('end string')
        

