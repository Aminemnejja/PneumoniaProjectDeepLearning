from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.config import Config

def compile_model(model, learning_rate=1e-4):
    """
    Compile le modèle avec l'optimiseur Adam et la perte binaire.
    """
    model.compile(
        optimizer=Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

def get_callbacks():
    """
    Retourne les callbacks nécessaires pour l'entraînement.
    """
    checkpoint = ModelCheckpoint(
        filepath=Config.CHECKPOINT_PATH,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True
    )
    return [checkpoint, early_stopping]
