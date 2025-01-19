import config
from dataset import MyDataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from model import SimpleYOLO
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import train_model, evaluate_model, calculate_loss
import import_ipynb
from preprocessing import train_loader, val_loader


num_classes = 2
class_to_idx = {'dog': 0, 'cat':1}

model = SimpleYOLO(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, optimizer, num_epochs, device, num_classes, class_to_idx
)