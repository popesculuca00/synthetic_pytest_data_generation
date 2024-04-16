import torch

def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    r
    max_seq_length = sequence.batch_sizes.size(0)
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError("Expected total_length to be at least the length "
                             "of the longest sequence in input, but got "
                             "total_length={} and max sequence length being {}"
                             .format(total_length, max_seq_length))
        max_seq_length = total_length
    padded_output, lengths = torch._C._VariableFunctions._pad_packed_sequence(
        sequence.data, sequence.batch_sizes, batch_first, padding_value, max_seq_length)
    if sequence.unsorted_indices is not None:
        batch_dim = 0 if batch_first else 1
        return padded_output.index_select(batch_dim, sequence.unsorted_indices), \
            lengths[sequence.unsorted_indices]
    return padded_output, lengths