from data import CIFARDataset
from models import SimpleCNN

import torch
from torch import nn

TRAIN_EPOCHS = 100
BATCH_SIZE = 32

def main():
    data = CIFARDataset()
    model = SimpleCNN()
    
    # create train/val split
    train_data, val_data = torch.utils.data.random_split(data, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for _ in range(TRAIN_EPOCHS):
        # training loop
        total_loss = 0 
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        print(f'train loss: {total_loss}')
        
        # validation loop
        total_loss = 0
        total_correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                preds = model(inputs)
                loss = loss_fn(preds, labels)
                pred_classes = torch.argmax(preds, 1)
                total_correct += torch.sum(pred_classes == labels)
                total_loss += loss.item()

        print(f'validation loss: {total_loss}')
        print(f'validation accuracy: {total_correct / len(val_data)}')
        
if __name__ == '__main__':
    main()