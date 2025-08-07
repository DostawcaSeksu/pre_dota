import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv('data/dota_preprocessed_data.csv', index_col='match_id')
x = df.drop('radiant_win', axis=1).values
y = df['radiant_win'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class DotaPredictor(nn.Module):
    def __init__(self, input_features):
        super(DotaPredictor, self).__init__()
        self.layer1 = nn.Linear(input_features, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x))
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing {device} device')

input_size = x_train_scaled.shape[1]
model = DotaPredictor(input_features=input_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())

        acc = accuracy_score(y_test, all_preds)
        avg_loss = total_loss / len(train_loader)
        print(f'epoch [{epoch+1}/{epochs}], loss: {avg_loss:.4f}, test accuracy: {acc:.2%}')

print('\n---Final score---')
print(f'PyTorch model Accuracy: {accuracy_score(y_test, all_preds):.2%}')
print('\nDetailed classification report:')
print(classification_report(y_test, all_preds, target_names=['Wire Win', 'Radiant Win']))