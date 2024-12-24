# utils/callbacks.py

import torch
import os

def save_checkpoint(model, epoch, optimizer, loss, checkpoint_dir, filename="checkpoint.pth"):
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
