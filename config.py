from datetime import datetime
import os
# Répertoires de données
BASE_DIR = "C:\\Users\menaj\\.cache\\kagglehub\\datasets\\paultimothymooney\\chest-xray-pneumonia\\versions\\2\\chest_xray"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Paramètres du modèle
INPUT_SHAPE = (3, 224, 224)  # Format (canal, hauteur, largeur)
BATCH_SIZE = 32
EPOCHS = 10

# Optimiseur et loss
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'
METRICS = ['accuracy']

# Appareil de calcul
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)