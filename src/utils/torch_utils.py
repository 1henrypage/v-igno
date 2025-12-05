import torch

def get_default_device() -> torch.device:
    """ Returns GPU if available, else CPU. """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")