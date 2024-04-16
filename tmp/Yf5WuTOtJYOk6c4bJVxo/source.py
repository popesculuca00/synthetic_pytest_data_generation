def train_batch(model, x, target, optimizer, criterion):
    

    # Forward
    outputs = model(x)

    # Loss computation
    batch_loss = criterion(outputs, target)

    # Backprop
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()