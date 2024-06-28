import torch
from torch import nn
from torch.nn import functional as F


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

    def __init__(
        self, hidden_size, num_classes, n_feats, num_layers, dropout, input_size
    ):
        super(SpeechRecognition, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = ModelRNN(n_feats, hidden_size)  # Utilisation du RNN personnalisé
        self.input_size = input_size
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, self.input_size),
            nn.LayerNorm(self.input_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.input_size, self.input_size),
            nn.LayerNorm(self.input_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=self.input_size,
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
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.final_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden):
        out, (hn, cn) = self.lstm(x, hidden)
        x = self.dropout(
            F.gelu(self.layer_norm(out))
        )  # Shape: batch_size, time_steps, hidden_size
        return self.final_fc(x), (hn, cn)


def accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs, _ = model(inputs.unsqueeze(1), model._init_hidden(inputs.size(0)))
            _, predicted = torch.max(outputs.squeeze(0).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
