import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from custom_preprocessing import Custom_preprocessing

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("using ", device, " device...")

# Importation de la classe mere


def load_hyper_parameters(src_file="dataset/model_parameters.json"):
    if not os.path.exists(src_file):
        print(
            "Le fichier de parametre du modele est introvable, vous devez la recreer!"
        )
        return {}

    with open(src_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data


class ModelRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


class SpeechRecognition(nn.Module):
    hyper_parameters = {
        "num_classes": 29,
        "n_feats": 1131,
        "dropout": 0.1,
        "hidden_size": 1024,
        "num_layers": 1,
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
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bidirectional=False,
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def _init_hidden(self, batch_size):
        n, hs = self.num_layers, self.hidden_size
        return (torch.zeros(n * 1, batch_size, hs), torch.zeros(n * 1, batch_size, hs))

    def forward(self, x, hidden):
        x = x.squeeze(1)  # batch, feature, time
        x = self.rnn(x)  # batch, time, hidden_size
        x = self.dense(x)  # batch, time, feature
        x = x.transpose(0, 1)  # time, batch, feature
        out, (hn, cn) = self.lstm(x, hidden)
        # Ajustement des dimensions de hx et cx
        hn = hn.squeeze(0)  # hn devient 2D
        cn = cn.squeeze(0)  # cn devient 2D
        x = self.dropout2(F.gelu(self.layer_norm2(out)))  # (time, batch, n_class)
        return self.final_fc(x), (hn, cn)


# pretraitement de l'audio a predire
custom = Custom_preprocessing("./0_george_1.wav", "decode_test.npy")

hyper_parameters = load_hyper_parameters()
hidden_size = hyper_parameters["hidden_size"]
num_classes = hyper_parameters["num_classes"]
n_feats = hyper_parameters["input_size"]
num_layers = 1
dropout = 0.1
model = SpeechRecognition(
    hidden_size=hidden_size,
    num_classes=num_classes,
    n_feats=n_feats,
    num_layers=num_layers,
    dropout=dropout,
).to(device)
model.load_state_dict(torch.load("./model1.pth"))
model.eval()

# Chargement du donnee pretraitee
to_decode = np.load("decode_test.npy")
to_decode = to_decode.reshape(to_decode.shape[0], 1, to_decode.shape[1])

with torch.no_grad():
    x = torch.tensor(to_decode)
    pred, _ = model(
        x.unsqueeze(1), model._init_hidden(x.size(0))
    )  # Ajout de unsqueeze(1) pour correspondre à la taille du batch
    y = F.softmax(pred, dim=-1)
    print(y, y.shape)
