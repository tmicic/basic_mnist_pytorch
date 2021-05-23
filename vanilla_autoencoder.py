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




TRAIN_BATCH_SIZE = 32
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
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.encoding = nn.Linear(9216, encoding_size)
        self.fc = nn.Linear(encoding_size, 28 * 28)

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
        x = x.view(-1,1,28,28)
        output = torch.sigmoid(x)
        return output, enc, self.conv1.weight

model = Model(encoding_size=ENCODING_SIZE).to(device=device)
optimizer = optim.Adadelta(model.parameters(), lr=LR)
loss_function = torch.nn.BCELoss(reduction='sum').to(device=device)

if MODEL_PATH != '' and LOAD_AND_CHECK:
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

if __name__ == '__main__':

    best_validate_loss = inf
    last_monitored_weights = None

    for epoch in range(0, EPOCHS + 1):
    
        if LOAD_AND_CHECK and epoch == 0:
            print(f'Evaluating model: {MODEL_PATH}')
        elif epoch==0:
            continue

        if epoch > 0:
            print(f'Epoch {epoch} of {EPOCHS}: ')

        for is_training, dataloader, context in [(True, train_dataloader, contextlib.nullcontext()), (False, validate_dataloader, torch.no_grad())]:

            if LOAD_AND_CHECK and epoch==0 and is_training:
                continue
            
            if is_training:
                print('\tTraining:')

                model.train()

            else:   # validating
                print('\tValidating:')
        
                model.eval()

            with context:

                total_loss = 0.
                
                grounds, preds = None, None

                for i, (X, y) in enumerate(dataloader):

                    X = X.to(device)
                    y = y.to(device)

                    if is_training: optimizer.zero_grad()

                    output, encoding, monitored_weights = model(X)

                    loss = loss_function(output, X)
                    
                    if is_training:
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()




                    print(f'\tBatch: {i+1:>6}/{len(dataloader):<6}\tLoss:{total_loss:.4f}', end='\r')

                print()

                if not is_training:
                    # if evaluating - show last outputs
                    plt.imshow(torch.cat([X.view(-1,28), output.view(-1,28)], dim=1).detach().cpu())
                    plt.show(block=False)
                    plt.pause(0.001)
                    


                if total_loss < best_validate_loss and not is_training:
                    if LOAD_AND_CHECK and epoch==0:
                        best_validate_loss = total_loss
                        print(f'\tModel has a loss of {total_loss:.4f} before further training.')
                    else:
                        print(f'\tNew best loss {total_loss:.4f} (previously {best_validate_loss:.4f}). Saving model...', end='')
                        torch.save(model.state_dict(), MODEL_PATH)
                        best_validate_loss = total_loss
                    print()




        print()
        