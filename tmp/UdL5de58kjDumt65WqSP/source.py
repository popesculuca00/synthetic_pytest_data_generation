def calculate_accuracy(y_pred, y):
    
    acc = ((y_pred.argmax(dim=1) == y).float().mean())
    return acc