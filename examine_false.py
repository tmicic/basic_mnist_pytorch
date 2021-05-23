import torch

train_encodings = torch.load('train_encodings.torch')
train_labels = torch.load('train_labels.torch')
test_encodings = torch.load('test_encoding.torch')
test_labels = torch.load('test_labels.torch')

top_k = 100
dists = torch.cdist(train_encodings, test_encodings,)
argmins = torch.argmin(dists, dim=0)

def red(text):
    print('\033[31m', text, '\033[0m', sep='')

for n in range(test_encodings.shape[0]):

    
    
    if train_labels[argmins[n]] != test_labels[n]:
        red(' Wrong Prediction')
        print(f'Assessing prediction number {n}')
        print(f' Ground Truth: {test_labels[n]}')
        print(f' Top Prediction: {train_labels[argmins[n]]}')

        d, i = torch.sort(dists[:,n])
        
        p = train_labels[i]

        print(d[:top_k])
        print(p[:top_k])

        print()










