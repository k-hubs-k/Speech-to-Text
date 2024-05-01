import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loading import X, y
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Diviser X et y en un ensemble d'entraînement (80%) et un ensemble de validation (20%)
# X = np.load("mfcc_features.npy")
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1000
)

# Définir les hyperparamètres
input_size = X_train.shape[1]  # Nombre de coefficients MFCC
hidden_size = 64  # Taille de la couche cachée
num_classes = y.shape[0]  # Nombre de classes (mots/phrases)
num_epochs = 500
batch_size = len(y_train)
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# mfcc_features = np.random.rand(100, 13)
# labels = np.random.randint(0, 10, size=100)

# Convertir les données en tensors PyTorch
X_train_tensor = torch.Tensor(X_train)
X_test_tensor = torch.Tensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Créer des DataLoader PyTorch
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(X_train_tensor.shape)


# Définir l'architecture du modèle
class SpeechToTextModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SpeechToTextModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def accuracy(proba_batch, label_batch):
    correct = 0
    preds = torch.argmax(proba_batch, dim=1)
    for i, pred in enumerate(preds):
        if pred == label_batch[i]:
            correct += 1
    return correct / batch_size


# Initialiser le modèle
model = SpeechToTextModel(input_size, hidden_size, num_classes)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entraîner le modèle
loss1 = []
l = []
for epoch in tqdm(range(num_epochs)):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc = accuracy(outputs, labels)
    loss1.append(loss.item())
    loss, acc = 0.0, 0.0
    batch = 0
    for inputs, labels in test_loader:
        batch += 1
        with torch.no_grad():
            outputs = model(inputs)
            loss += criterion(outputs, labels)
            acc += accuracy(outputs, labels)
    loss /= batch
    acc /= batch
    l.append(loss.item())
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # print(f'valid_loss = {loss : .3} \nvalid_acc = {acc : .3}')
print(f"valid_loss = {loss : .3} \nvalid_acc = {acc : .3}")
"""loss = 0.
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    loss = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss.append(criterion(outputs, labels).item())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy du neurone sur le test set: {100 * correct / total}%')
    
print('Entraînement terminé.')
print(f'loss accuracy : {loss[len(loss) - 1]}')"""

# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), "model.pth")
