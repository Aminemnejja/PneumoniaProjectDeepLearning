from models.cnn_model import create_simple_cnn
from models.transfer_learning import create_transfer_learning_model

def get_model(model_type="simple", input_shape=(224, 224, 3), num_classes=2):
    """
    Retourne un modèle basé sur le type spécifié.
    """
    if model_type == "simple":
        return create_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    elif model_type == "transfer_learning":
        return create_transfer_learning_model(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError("Modèle non reconnu. Choisissez entre 'simple' et 'transfer_learning'.")
