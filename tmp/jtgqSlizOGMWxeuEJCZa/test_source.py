# -*- coding: utf-8 -*-

import pytest
import torch

from source import one_hot

class TestOneHot:

    def test_one_hot(self):
        # Given
        indices = torch.LongTensor([1, 0, 2])
        depth = 3

        # When
        result = one_hot(indices, depth)

        # Then
        expected_result = torch.tensor([[0., 1., 0.],
                                        [1., 0., 0.],
                                        [0., 0., 1.]])
        assert torch.allclose(result, expected_result)

    def test_one_hot_with_cuda(self):
        # Given
        indices = torch.LongTensor([1, 0, 2]).cuda()
        depth = 3

        # When
        result = one_hot(indices, depth)

        # Then
        expected_result = torch.tensor([[0., 1., 0.],
                                        [1., 0., 0.],
                                        [0., 0., 1.]], device='cuda')
        assert torch.allclose(result, expected_result)

if __name__ == "__main__":
    pytest.main()