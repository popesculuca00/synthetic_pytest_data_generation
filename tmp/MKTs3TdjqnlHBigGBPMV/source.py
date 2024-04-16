def sequence_mask(seq_ids, valid_lengths):
    
    lengths_exp = valid_lengths.unsqueeze(1)
    mask = seq_ids < lengths_exp

    return mask