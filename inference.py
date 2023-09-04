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

myModel = torch.load("./checkpoint/ppai_weight.pkl")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
next(myModel.parameters()).device

def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)

            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)

            _, prediction = torch.max(outputs, 1)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    #write json

inference(myModel, val_dl)