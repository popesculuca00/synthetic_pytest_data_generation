def captum_sequence_forward(inputs, attention_mask=None, position=0, model=None):
    
    model.eval()
    model.zero_grad()
    pred = model(inputs, attention_mask=attention_mask)
    pred = pred[position]
    return pred