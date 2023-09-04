import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from SoundDS import SoundDs
from model import AudioClassifier
import atexit

df = pd.read_csv("./dataset/dataset.csv", header=None)

myds = SoundDs(df, "./dataset/dataset.csv")

num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

train_dl = DataLoader(train_ds, batch_size=16)
val_dl = DataLoader(train_ds, batch_size=16)

print("If you forget backup your checkpoint, please do it before input.")

checkpoint = input("Use checkpoint?[Y(yes)](press Enter to no)")
if input == "Y" or input == "yes":
    myModel = torch.load("./checkpoint/ppai_weight.pkl")
else:
    myModel = AudioClassifier()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
next(myModel.parameters()).device

@atexit.register
def q():
    torch.save(myModel, "./checkpoint/ppai_weight.pkl")

def training(model, train_dl, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)), epochs=num_epochs, anneal_strategy='linear')

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)

            correct_prediction += (prediction == labels).sum().item()

            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuray: {acc:.2f}')
    
    print('finished')

num_epochs = 10000
training(myModel, train_dl, num_epochs)