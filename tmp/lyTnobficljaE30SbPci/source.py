def train_batch(model, x, target, optimizer, criterion):
    

    # Forward
    outputs = model(x)

    # Loss computation
    batch_loss = criterion(outputs, target)
    pred = outputs.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum()

    # Backprop
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss, correct