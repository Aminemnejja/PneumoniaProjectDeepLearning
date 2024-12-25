import matplotlib.pyplot as plt

def plot_history(history):
    """
    Affiche les courbes de précision et de perte pour l'entraînement et la validation.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Précision d\'entrainement')
    plt.plot(history.history['val_accuracy'], label='Précision de validation')
    plt.title('Précision')
    plt.xlabel('Epochs')
    plt.ylabel('Précision')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perte d\'entrainement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.title('Perte')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_model_performance(history, model_name="model"):
    """
    Sauvegarde les courbes d'entrainement et de validation dans un fichier.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Précision d\'entrainement')
    plt.plot(history.history['val_accuracy'], label='Précision de validation')
    plt.title('Précision')
    plt.xlabel('Epochs')
    plt.ylabel('Précision')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perte d\'entrainement')
    plt.plot(history.history['val_loss'], label='Perte de validation')
    plt.title('Perte')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}_performance.png')
    plt.close()
