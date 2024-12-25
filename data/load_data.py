import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    """
    Charge les données depuis le répertoire et les prépare pour l'entraînement et la validation.
    """
    # Augmentation des données
    datagen = ImageDataGenerator(
        rescale=1./255,            # Normalisation
        rotation_range=20,        # Rotation aléatoire
        width_shift_range=0.2,    # Décalage horizontal
        height_shift_range=0.2,   # Décalage vertical
        shear_range=0.2,          # Cisaillement
        zoom_range=0.2,           # Zoom
        horizontal_flip=True,     # Retourner horizontalement
        fill_mode='nearest'       # Remplir les pixels manquants après transformation
    )

    # Chargement des données d'entraînement et de validation
    train_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_generator, validation_generator
