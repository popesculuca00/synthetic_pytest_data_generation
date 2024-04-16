def attention_padding_mask(q, k, padding_index=0):
    

    mask = k.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
    return mask