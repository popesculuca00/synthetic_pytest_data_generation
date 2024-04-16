import torch

def _normalize_images(images):
    
    # Shift the image from [-1, 1] range to [0, 1] range.
    min_val = float(images.min())
    max_val = float(images.max())
    images.clamp_(min=min_val, max=max_val)
    images.add_(-min_val).div_(max_val - min_val + 1e-5)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    images = images.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(
        'cpu', torch.uint8).numpy()

    return images