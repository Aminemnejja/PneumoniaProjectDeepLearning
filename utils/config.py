class Config:
    DATA_DIR = '/mnt/c/Users/menaj/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DROPOUT_RATE = 0.5
    CHECKPOINT_PATH = './model_checkpoints/model_best.keras'
    EARLY_STOPPING_PATIENCE = 5
