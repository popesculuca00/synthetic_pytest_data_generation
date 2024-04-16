import pytest
import torch

# Import the source code
from source import bboxes_iou

class TestBboxesIou:

    def test_bboxes_iou(self):
        # Test case 1
        bboxes_a = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
        bboxes_b = torch.tensor([[5, 5, 15, 15]])
        expected_output = torch.tensor([0.1])
        assert torch.allclose(bboxes_iou(bboxes_a, bboxes_b), expected_output)

        # Test case 2
        bboxes_a = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
        bboxes_b = torch.tensor([[5, 5, 15, 20]])
        expected_output = torch.tensor([0.25])
        assert torch.allclose(bboxes_iou(bboxes_a, bboxes_b), expected_output)

        # Test case 3
        bboxes_a = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
        bboxes_b = torch.tensor([[5, 5, 20, 20]])
        expected_output = torch.tensor([0.25])
        assert torch.allclose(bboxes_iou(bboxes_a, bboxes_b), expected_output)

        # Test case 4
        bboxes_a = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
        bboxes_b = torch.tensor([[5, 5, 15, 15], [5, 5, 20, 20]])
        expected_output = torch.tensor([0.1, 0.25])
        assert torch.allclose(bboxes_iou(bboxes_a, bboxes_b), expected_output)

pytest.main()