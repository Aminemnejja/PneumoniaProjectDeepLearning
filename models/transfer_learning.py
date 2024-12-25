from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def create_transfer_learning_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Crée un modèle avec transfert de learning utilisant VGG16.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False  # Gel des poids de la base VGG16

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model
