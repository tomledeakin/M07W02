import pandas as pd
import torch
from torchvision import transforms
from dataset import ImageDataset
import config
import os
from sklearn.model_selection import train_test_split
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader

annotations_dir = os.path.join(config.data_dir, 'annotations')
images_dir = os.path.join(config.data_dir, 'images')

image_files = [file for file in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, file))]
annotation_files = [annot for annot in os.listdir(annotations_dir) if os.path.isfile(os.path.join(annotations_dir, annot))]

df = pd.DataFrame({'image_name': image_files})
print(df.head())


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# dataset
train_dataset = ImageDataset(annotations_dir, images_dir, transform=transform)
val_dataset = ImageDataset(annotations_dir, images_dir, transform=transform)

train_dataset.image_files = [file for file in train_dataset.image_files if file in train_df['image_name'].values]
val_dataset.image_files = [file for file in val_dataset.image_files if file in val_df['image_name'].values]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
print(num_features)
model.fc = nn.Linear(num_features, 2)

model = model.to(config.device)
print(f'device: {config.device}')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(config.device)
        label = label.to(config.device)

        prediction = model(data)
        loss = criterion(prediction, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, label in val_loader:
            data = data.to(config.device)
            label = label.to(config.device)
            prediction = model(data)
            _, predicted = torch.max(prediction, 1)
            correct += (predicted==label).sum()
            total += label.size(0)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {float(correct) / float(total) * 100:.2f}%")


# Final Result

# Epoch 1/10, Validation Accuracy: 86.18%
# Epoch 2/10, Validation Accuracy: 91.87%
# Epoch 3/10, Validation Accuracy: 92.68%
# Epoch 4/10, Validation Accuracy: 92.95%
# Epoch 5/10, Validation Accuracy: 93.50%
# Epoch 6/10, Validation Accuracy: 90.11%
# Epoch 7/10, Validation Accuracy: 92.41%
# Epoch 8/10, Validation Accuracy: 96.61%
# Epoch 9/10, Validation Accuracy: 95.39%
# Epoch 10/10, Validation Accuracy: 95.66%
