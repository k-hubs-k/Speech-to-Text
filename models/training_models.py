import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.

# Diviser X et y en un ensemble d'entraînement (80%) et un ensemble de validation (20%)
X = np.load("./models/mfcc_feature1.npy")
y = np.load("./models/labels.npy")

X_train, X_set, y_train, y_set = train_test_split(X, y, test_size=0.2, random_state=34)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_set, y_set, test_size=0.5, random_state=34
)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_valid = X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# Définir les hyperparamètres
input_size = X_train.shape[2]  # Nombre de coefficients MFCC
hidden_size = 1024  # Taille de la couche cachée
num_classes = y.shape[0]  # Nombre de classes (mots/phrases)
num_epochs = 120
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = len(X_train)

# Convertir les données en tensors PyTorch
X_train_tensor = torch.Tensor(X_train)
X_valid_tensor = torch.Tensor(X_valid)
X_test_tensor = torch.Tensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_valid_tensor = torch.LongTensor(y_valid)
y_test_tensor = torch.LongTensor(y_test)

# Créer des DataLoader PyTorch
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(X_valid), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=True)


def get_training_data():
    print("Getting train/test/validation data...")
    out = {
        "X": X,
        "y": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
    }
    return out


#######################
###### TRAINING #######
#######################

# Définition du modèle
model = SpeechRecognition(
    hidden_size=hidden_size,
    num_classes=num_classes,
    n_feats=input_size,
    num_layers=1,
    dropout=0.1,
)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss = []
train_acc = []
test_loss = []
test_acc = []
best_train_acc = 0
best_test_acc = 0
# Entraînement du modèle
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(
            inputs.unsqueeze(1), model._init_hidden(inputs.size(0))
        )  # Ajout de unsqueeze(1) pour correspondre à la taille du batch
        loss = criterion(
            outputs.squeeze(), labels
        )  # Ajustement pour correspondre à la taille du batch
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        acc = accuracy(train_loader, model)

    epoch_loss = running_loss / len(train_loader)
    train_loss.append(epoch_loss)
    train_acc.append(acc)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs.unsqueeze(1), model._init_hidden(inputs.size(0)))
            loss = criterion(outputs.squeeze(), labels)
            _, predicted = torch.max(outputs.squeeze(0).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc1 = correct / total
    test_loss.append(loss)
    test_acc.append(acc1)

    # record best train and test
    if acc > best_train_acc:
        best_train_acc = acc
    if acc1 > best_test_acc:
        best_test_acc = acc1

print(
    f"valid_loss = {100 * loss / total : .3}% \nvalid_acc = {100 * best_test_acc : .3}%"
)
print("Entraînement terminé.")

# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), "./models/model.pth")
