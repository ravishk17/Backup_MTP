import torch
# 6. Save and Load Model
def save_model(model, path):
    """Save the trained model to a file."""
    torch.save(model.state_dict(), path)

def load_model(model_class, path, *args, **kwargs):
    """Load a model from a file."""
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
