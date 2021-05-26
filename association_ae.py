# adapted from: https://github.com/pytorch/examples/tree/master/mnist
# save and load model: https://pytorch.org/tutorials/beginner/saving_loading_models.html

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
import tqdm     # https://www.youtube.com/watch?v=RKHopFfbPao&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=50


TRAIN_BATCH_SIZE = 32
TRAIN_SHUFFLE = False
NUM_WORKERS = 4
PIN_MEMORY = True
USE_CUDA = True
VALIDATE_BATCH_SIZE = 512
VALIDATE_SHUFFLE = False
ENCODING_SIZE = 128

MODEL_PATH = 'mnist.model'
LOAD_AND_CHECK = False

LR = 0.06
EPOCHS = 10

ensure_reproducibility(6000)

device = torch.device("cuda" if USE_CUDA else "cpu")

train_transform = transforms.Compose([transforms.ToTensor()])#,transforms.Normalize((0.1307,), (0.3081,))])
test_transform = transforms.Compose([transforms.ToTensor()])#, transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = dataset.FashionMNIST(download=True, root='data', train=True, transform=train_transform)
validate_dataset = dataset.FashionMNIST(download=True, root='data', train=False, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=TRAIN_SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)
validate_dataloader = DataLoader(validate_dataset, batch_size=VALIDATE_BATCH_SIZE, shuffle=VALIDATE_SHUFFLE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, worker_init_fn=seed_worker)

class Model(nn.Module):
    def __init__(self, encoding_size=128):
        super().__init__()
        self.encoding_size = encoding_size
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.encoding = nn.LazyLinear(encoding_size)
        self.fc = nn.Linear(encoding_size, 28 * 28)

    def get_encoding(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        #x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        enc = self.encoding(x)

        return enc

    def forward(self, x):
        enc = self.get_encoding(x)
        x = F.relu(enc)
        x = self.fc(x)
        x = x.view(-1,1,28,28)
        return x, enc, self.conv1.weight

model = Model(encoding_size=ENCODING_SIZE).to(device=device)
optimizer = optim.Adadelta(model.parameters(), lr=LR)
optimizer = optim.Adadelta(list(model.conv1.parameters())+list(model.conv2.parameters()), lr=LR)
loss_function = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device=device)

# model.fc.requires_grad_(False)
# model.encoding.requires_grad_(False)

# MIXED PRECISION TRAINING (using FP16 instead, increase memory and apparently speed)
# https://www.youtube.com/watch?v=ks3oZ7Va8HU&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=49
#
scaler = torch.cuda.amp.grad_scaler.GradScaler()


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

        training_encodings = None
        training_labels = None

        for is_training, dataloader, context in [(True, train_dataloader, contextlib.nullcontext()), (False, validate_dataloader, torch.no_grad())]:

            if LOAD_AND_CHECK and epoch==0 and is_training:
                continue
            
            if is_training:
                print('\tTraining:')

                training_encodings = None
                training_labels = None
                model.train()

            else:   # validating
                print('\tValidating:')
        
                model.eval()

            with context:

                total_loss = 0.
                total_correct = 0
                total_processed = 0
                
                grounds, preds = None, None

                for i, (X, y) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), leave=False):

                    X = X.to(device)
                    y = y.to(device)

                    if is_training: optimizer.zero_grad()

                    with torch.cuda.amp.autocast_mode.autocast():
                        output, encoding, _ = model(X)

                        if is_training:
                            if training_encodings is None:
                                training_encodings = encoding.detach()
                                training_labels = y.detach()
                            else:
                                training_encodings = torch.cat([training_encodings, encoding.detach()])
                                training_labels = torch.cat([training_labels, y.detach()])

                        loss = loss_function(output, X)
                        
                    if is_training:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    total_loss += loss.item()
                    total_processed += len(y)

                    if not is_training:
                        dists = torch.cdist(training_encodings, encoding)
                        torch.save(training_encodings, 'train_encodings.torch')
                        torch.save(training_labels, 'train_labels.torch')
                        torch.save(encoding, 'test_encoding.torch')
                        torch.save(y, 'test_labels.torch')

                        best_matches_with_train_set = training_labels[torch.argmin(torch.cdist(training_encodings, encoding), dim=0)]
                        n_correct = torch.sum(best_matches_with_train_set==y)
                        total_correct += n_correct

                        # incorrect = best_matches_with_train_set!=y

                        # print(y[incorrect])
                        # print(best_matches_with_train_set[incorrect])
                        # exit()



                    #print(f'\033[K\tBatch: {i+1:>6}/{len(dataloader):<6}\tLoss:{total_loss:.4f}\tAcc: {total_correct/total_processed*100:.2f}%', end='\r')

                print()

                if not is_training:
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
        