# main.py

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import config
from utils.data_loader import load_data
from utils.callbacks import save_checkpoint
from utils.visualisation import plot_confusion_matrix, plot_training_history
from models.simple_cnn import SimpleCNN
from models.mobilenet import MobileNet

# Charger les données
train_loader, val_loader, test_loader = load_data(config.TRAIN_DIR, config.VALIDATION_DIR, config.TEST_DIR, batch_size=config.BATCH_SIZE)

# Sélectionner le modèle
model_name = "simple_cnn"  # Ou "mobilenet"
if model_name == "simple_cnn":
    model = SimpleCNN().to(config.DEVICE)
else:
    model = MobileNet().to(config.DEVICE)

# Optimiseur et loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Entraînement du modèle
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
for epoch in range(config.EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item()

            predicted = (outputs > 0.5).float()
            val_total += labels.size(0)
            val_correct += (predicted.squeeze() == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_correct / val_total)

    # Sauvegarde des checkpoints
    save_checkpoint(model, epoch, optimizer, loss.item(), config.CHECKPOINT_DIR)

    print(f"Epoch {epoch + 1}/{config.EPOCHS}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, "
          f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_correct / val_total:.4f}")

# Visualisation des résultats
plot_confusion_matrix(val_accuracies, train_accuracies, "Confusion Matrix")
plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
