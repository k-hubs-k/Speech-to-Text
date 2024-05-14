from preprocessing import remove_silence, normalize_audio_volume, remove_background_noise, remove_artifacts, temporal_normalization, convert_to_spectrogram, extract_features_mfcc, target_length
from SpeechRecognition import SpeechRecognition, hidden_size, X_valid, input_size, num_epochs, valid_loader
import numpy as np
import torch                 

# Accuracy
# Charger le modèle depuis le fichier 'model.pth'
model = SpeechRecognition(hidden_size=hidden_size, num_classes=len(X_valid), n_feats=input_size, num_layers=1, dropout=0.1)
# model.load_state_dict(torch.load('model.pth'))
model.eval()

# Définir le mode d'évaluation sur le GPU s'il est disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

acc = []
for epoch in range(num_epochs):
    # Évaluer le modèle sur l'ensemble de validation
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs.unsqueeze(1), model._init_hidden(inputs.size(0)))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    acc.append(accuracy)
print(f"Accuracy on validation set: {100 * accuracy:.2f}%")

# Prétraitement et conversion de l'audio en caractéristiques
mfcc_features_list = []
remove_silence("9922809_782992.wav", "scripts/tmp/output_audio_without_silence.wav")
normalize_audio_volume("scripts/tmp/output_audio_without_silence.wav","scripts/tmp/volume_normalized_audio.wav")
remove_background_noise(
    "scripts/tmp/volume_normalized_audio.wav", "scripts/tmp/supp_bruit.wav"
)
remove_artifacts("scripts/tmp/supp_bruit.wav", "scripts/tmp/output_audio_filtered.wav")
temporal_normalization(
    "scripts/tmp/output_audio_filtered.wav", "scripts/tmp/temporal_normalization.wav", target_length
)
convert_to_spectrogram("scripts/tmp/temporal_normalization.wav", "scripts/tmp/spectrogram.npy")
mfcc_features = extract_features_mfcc("scripts/tmp/temporal_normalization.wav")
# Normalisation des caractéristiques MFCC (par exemple, en mettant à l'échelle entre -1 et 1)
mfcc_features_normalized = (mfcc_features - np.mean(mfcc_features)) / np.std(
    mfcc_features
)
mfcc_features_list.append(mfcc_features_normalized)
# Convertir la liste en un tableau numpy
X = np.array(mfcc_features_list)
X = torch.Tensor(X)
X = X.reshape(X.shape[0], 1, -1)
print(X.shape)

# Passage des caractéristiques au modèle
model.eval()  # Mettre le modèle en mode évaluation
print(X.unsqueeze(0).shape)
outputs, _ = model(X.unsqueeze(0), model._init_hidden(1))  # Supposant un batch de taille 1

class GreedyDecoder:
    def __init__(self, vocab):
        self.vocab = vocab  # Liste des caractères dans votre vocabulaire

    def decode(self, predicted_indices):
        decoded_text = []
        for idx in predicted_indices:
            if idx.item() == 0:  # Indice 0 pour le caractère de début [START]
                continue
            elif idx.item() == 1:  # Indice 1 pour le caractère de fin [END]
                break
            else:
                decoded_text.append(self.vocab[idx.item()])
        return ''.join(decoded_text)
    
# Définir le vocabulaire
vocab = ['<PAD>', '<START>', '<END>', '<UNK>', ' ', "'", ',', '.', '-', '!', '?',
         'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z']
# Instancier le décodeur
decoder = GreedyDecoder(vocab)

def postprocess_predictions(outputs, decoder):
    # Supprimer les symboles spéciaux et obtenir l'indice du caractère le plus probable pour chaque pas de temps
    predicted_indices = torch.argmax(outputs, dim=-1)
    predicted_text = decoder.decode(predicted_indices)  # Utilisez votre décodeur pour obtenir le texte
    
    # Supprimer les symboles spéciaux et nettoyer la transcription
    cleaned_text = predicted_text.replace("[START]", "").replace("[END]", "").replace("[UNK]", "").strip()
    
    return cleaned_text

# Utilisation de la fonction postprocess_predictions
predicted_text = postprocess_predictions(outputs, decoder)

# Affichage du résultat
print("Transcription audio prédite :", predicted_text)