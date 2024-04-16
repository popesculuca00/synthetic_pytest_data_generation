def convert_padding_mask_to_attention_mask(sequence, padding_mask):
    
    assert padding_mask.shape[0] == sequence.shape[0] and \
                                            'batch size mismatch between input sequence and  padding_mask'
    assert len(padding_mask.shape) == 2 and \
                                            'Can only convert 2D position mask to 3D attention mask'

    attention_mask = padding_mask[:, None, :].repeat(*(1, sequence.shape[1], 1))
    return attention_mask