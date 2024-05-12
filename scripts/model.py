import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from data_loading import X, y
from tqdm import tqdm

# Diviser X et y en un ensemble d'entraînement (80%) et un ensemble de validation (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

# Définir les hyperparamètres
input_size = X_train.shape[1]  # Nombre de coefficients MFCC
hidden_size = 64  # Taille de la couche cachée
num_classes = y.shape[0]  # Nombre de classes (mots/phrases)
num_epochs = 1000
batch_size = len(X_train)
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#mfcc_features = np.random.rand(100, 13)
#labels = np.random.randint(0, 10, size=100)

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

# Définir l'architecture du modèle
class SpeechToTextModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SpeechToTextModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)  # Dropout avec une probabilité de désactivation de 0.5
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        x = self.dropout(x)  # Appliquer la dropout après la couche de activation
        out = self.fc(out)
        return out


def accuracy(proba_batch, label_batch):
    correct = 0
    preds = torch.argmax(proba_batch, dim=1)
    for i, pred in enumerate(preds):
        if pred == label_batch[i]:
            correct += 1
    return correct / batch_size

# Initialiser le modèle
model = SpeechToTextModel(input_size, hidden_size, 2, num_classes)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001) # Utilisation du paramètre weight_decay pour la régularisation L2

# Entraîner le modèle
loss1 = []
loss_acc = []
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
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        loss_acc.append(loss)
print(f'valid_loss = {100 * loss / total : .3}% \nvalid_acc = {100 * correct / total : .3}%')
#print(f'Accuracy du neurone sur le test set: {100 * correct / total}%')

print('Entraînement terminé.')
# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), 'model.pth')