import torch
import numpy as np

class CIFARDataset(torch.utils.data.Dataset):
    def __init__(self):
        cifar = unpickle('cifar-10-batches-py/data_batch_1')
        self.data = [torch.tensor(x).reshape([32, 32, 3]).swapaxes(0, 2).swapaxes(1, 2).float() for x in cifar[b'data']]
        self.labels = [torch.tensor(x).long() for x in cifar[b'labels']]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    print('hello')
    class_names = unpickle('cifar-10-batches-py/batches.meta')
    cifar = unpickle('cifar-10-batches-py/data_batch_1')
    print(class_names)
    data = cifar[b'data']
    labels = cifar[b'labels']
    print(labels)
    print(len(labels))
    print(data)
    print(len(data))
    
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    main()
