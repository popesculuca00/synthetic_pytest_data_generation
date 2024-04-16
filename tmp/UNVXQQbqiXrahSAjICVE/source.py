def dropout_mask(x, sz, dropout):
    
    return x.new(*sz).bernoulli_(1 - dropout) / (1 - dropout)