import torch

def load_checkpoint(model, filepath):
    state_dict = torch.load(filepath, weights_only=False)
    model.load_state_dict(state_dict) 
    return model
