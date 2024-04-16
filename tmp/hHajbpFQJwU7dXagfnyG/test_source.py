# Import required libraries
import pytest
import torch

# Import the source function to test
from .source import weighted_sigmoid_log_loss

# Test class
class TestWeightedSigmoidLogLoss:

    @pytest.fixture
    def inputs(self):
        positive_predictions = torch.tensor([[1.0, 0.2, 0.3], [0.9, 0.8, 0.7]])
        negative_predictions = torch.tensor([[0.1, 0.3, 0.2], [0.2, 0.1, 0.4]])
        candidate_predictions = torch.tensor([[0.5, 0.4, 0.3], [0.6, 0.5, 0.4]])
        weight = torch.tensor([0.5, 0.4])
        alpha = 1.0

        return positive_predictions, negative_predictions, candidate_predictions, weight, alpha

    def test_weighted_sigmoid_log_loss(self, inputs):
        positive_predictions, negative_predictions, candidate_predictions, weight, alpha = inputs

        loss, reg_loss = weighted_sigmoid_log_loss(positive_predictions, negative_predictions, candidate_predictions, weight, alpha)

        assert torch.isclose(loss.item(), -3.05263252).all(), "The loss is not calculated correctly"
        assert torch.isclose(reg_loss.item(), 1.0).all(), "The regularization loss is not calculated correctly"