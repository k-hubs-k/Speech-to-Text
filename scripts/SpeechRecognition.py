import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from data_loading import X, y

# Diviser X et y en un ensemble d'entraînement (80%) et un ensemble de validation (20%)
X_train, X_set, y_train, y_set = train_test_split(X, y, test_size=0.2, random_state=34)
X_valid, X_test, y_valid, y_test = train_test_split(X_set, y_set, test_size=0.5, random_state=34)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
# Définir les hyperparamètres
input_size = X_train.shape[2]  # Nombre de coefficients MFCC
hidden_size = 1024  # Taille de la couche cachée
num_classes = y.shape[0]  # Nombre de classes (mots/phrases)
num_epochs = 150
batch_size = len(X_train)
learning_rate = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Convertir les données en tensors PyTorch
X_train_tensor = torch.Tensor(X_train)
X_valid_tensor = torch.Tensor(X_valid)
y_train_tensor = torch.LongTensor(y_train)
y_valid_tensor = torch.LongTensor(y_valid)

print(X_train_tensor.shape)

# Créer des DataLoader PyTorch
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

class ActDropNormRNN(nn.Module):
    def __init__(self, n_feats, dropout, keep_shape=False):
        super(ActDropNormRNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_feats)
        self.keep_shape = keep_shape
    
    def forward(self, x):
        x = x.transpose(1, 2)
        # x = self.norm(self.dropout(F.gelu(x)))
        x = self.dropout(F.gelu(self.norm(x)))
        if self.keep_shape:
            return x.transpose(1, 2)
        else:
            return x


class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 29,
        "n_feats": 1131,
        "dropout": 0.1,
        "hidden_size": 1024,
        "num_layers": 1
    }

    def __init__(self, hidden_size, num_classes, n_feats, num_layers, dropout):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = ModelRNN(n_feats, hidden_size)  # Utilisation du RNN personnalisé
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.0,
                            bidirectional=False)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*1, batch_size, hs),
                torch.zeros(n*1, batch_size, hs))

    def forward(self, x, hidden):
        x = x.squeeze(1)  # batch, feature, time
        x = self.rnn(x)  # batch, time, hidden_size
        x = self.dense(x) # batch, time, feature
        x = x.transpose(0, 1) # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        # Ajustement des dimensions de hx et cx
        hn = hn.squeeze(0)  # hn devient 2D
        cn = cn.squeeze(0)  # cn devient 2D
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)

class ModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out

class ModelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(ModelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden):
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout(F.gelu(self.layer_norm(out)))  # Shape: batch_size, time_steps, hidden_size
        return self.final_fc(x), (hn, cn)

def accuracy(proba_batch, label_batch):
    correct = 0
    preds = torch.argmax(proba_batch, dim=1)
    for i, pred in enumerate(preds):
        if pred == label_batch[i]:
            correct += 1
    return correct / batch_size

# Définition du modèle
model = SpeechRecognition(hidden_size=hidden_size, num_classes=num_classes, n_feats=input_size, num_layers=1, dropout=0.1)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Entraînement du modèle
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs.unsqueeze(1), model._init_hidden(inputs.size(0)))  # Ajout de unsqueeze(1) pour correspondre à la taille du batch
        loss = criterion(outputs.squeeze(), labels)  # Ajustement pour correspondre à la taille du batch
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

print("Entraînement terminé.")
# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), 'model.pth')