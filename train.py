from models.model_factory import get_model
from models.model_utils import compile_model, get_callbacks
from data.load_data import load_data
from utils.config import Config
from utils.plot_utils import plot_history, save_model_performance

def main():
    # Chargement des données
    train_generator, validation_generator = load_data(Config.DATA_DIR, Config.IMG_SIZE, Config.BATCH_SIZE)

    # Choisir le modèle
    model = get_model(model_type="transfer_learning", input_shape=Config.IMG_SIZE + (3,))

    # Compiler le modèle
    compile_model(model, Config.LEARNING_RATE)

    # Callbacks pour l'entraînement
    callbacks = get_callbacks()

    # Entraînement
    history = model.fit(
        train_generator,
        epochs=Config.EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    # Visualisation des performances
    plot_history(history)
    save_model_performance(history, model_name="transfer_learning_model")

if __name__ == "__main__":
    main()
