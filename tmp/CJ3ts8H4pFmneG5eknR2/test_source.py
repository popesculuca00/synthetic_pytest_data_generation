import pytest
import torch
from source import pad_packed_sequence

def test_pad_packed_sequence():
    # Create a PackedSequence object
    sequence = torch.nn.utils.rnn.PackedSequence(torch.randn(5, 4), torch.LongTensor([4, 3, 2, 1, 0]))
    
    # Test with batch_first = True
    padded_output, lengths = pad_packed_sequence(sequence, batch_first=True)
    assert padded_output.shape == (5, 4) # Check the shape of padded_output
    assert lengths.tolist() == [4, 3, 2, 1, 0] # Check the length of each sequence

    # Test with batch_first = False
    padded_output, lengths = pad_packed_sequence(sequence, batch_first=False)
    assert padded_output.shape == (4, 5) # Check the shape of padded_output
    assert lengths.tolist() == [4, 3, 2, 1, 0] # Check the length of each sequence

    # Test with padding_value = 1.0
    padded_output, lengths = pad_packed_sequence(sequence, padding_value=1.0)
    assert torch.allclose(padded_output[0, 1:], 1.0) # Check the padded values

    # Test with total_length larger than max sequence length
    sequence = torch.nn.utils.rnn.PackedSequence(torch.randn(3, 5), torch.LongTensor([5, 4, 3]))
    padded_output, lengths = pad_packed_sequence(sequence, total_length=7)
    assert padded_output.shape == (7, 5) # Check the shape of padded_output
    assert lengths.tolist() == [5, 4, 3] # Check the length of each sequence

    # Test with unsorted_indices
    sequence = torch.nn.utils.rnn.PackedSequence(torch.randn(3, 5), torch.LongTensor([1, 0, 2]), torch.LongTensor([5, 4, 3]))
    padded_output, lengths = pad_packed_sequence(sequence)
    assert torch.allclose(padded_output[0, :], sequence.data[1, :]) # Check the padded values
    assert lengths.tolist() == [5, 4, 3] # Check the length of each sequence