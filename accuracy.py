import torch

from models.training_models import (
    accuracy,
    get_training_data,
    hidden_size,
    input_size,
    num_classes,
)

training_data = get_training_data()
test_loader = training_data["test_loader"]

# Charger le modèle entraîné s'il n'a pas déjà été chargé
model = SpeechRecognition(
    hidden_size=hidden_size,
    num_classes=num_classes,
    n_feats=input_size,
    num_layers=1,
    dropout=0.1,
)
model.load_state_dict(torch.load("model.pth"))

# Mettre le modèle en mode évaluation
model.eval()

# Calculer la précision sur l'ensemble de donnée de test
accuracy_test = accuracy(test_loader, model)
print(f"Précision sur l'ensemble de test : {100*accuracy_test:.2f}%")
