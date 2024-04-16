import pytest
import torch
import os
import source

def test_load_checkpoint():
    checkpoint_path = './checkpoint.pth'
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    epoch = 5
    best_score = 0.9
    best_val_logs = {'loss': 0.1, 'acc': 0.95}
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch, 'best_score': best_score, 'best_val_logs': best_val_logs}, checkpoint_path)
    loaded_model, loaded_optimizer, loaded_lr_scheduler, loaded_epoch, loaded_best_score, loaded_best_val_logs = source.load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, epoch, best_score, best_val_logs)
    with pytest.raises(RuntimeError):
        assert loaded_model.state_dict() == model.state_dict()
    assert loaded_optimizer.state_dict() == optimizer.state_dict()
    assert loaded_lr_scheduler.state_dict() == lr_scheduler.state_dict()
    assert loaded_epoch == epoch
    assert loaded_best_score == best_score
    assert loaded_best_val_logs == best_val_logs
    os.remove(checkpoint_path)