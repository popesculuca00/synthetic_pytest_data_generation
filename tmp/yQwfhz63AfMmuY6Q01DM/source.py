def loss_batch(model, loss_func, xb, yb, opt=None):
    
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item()