import torch

def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, epoch, best_score, best_val_logs):
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    best_score = checkpoint['best_score']
    best_val_logs = checkpoint['best_val_logs']

    return model, optimizer, lr_scheduler, epoch, best_score, best_val_logs