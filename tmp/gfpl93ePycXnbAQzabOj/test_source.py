import pytest
from source import tree_classification
from sklearn.datasets import load_iris

# Define the input parameters for the function
@pytest.fixture
def input_parameters():
    iris_data = load_iris()
    return iris_data.data, iris_data.data, iris_data.target, iris_data.target

# Define the test function
def test_tree_classification(input_parameters):
    training_set_features, testing_set_features, training_set_labels, testing_set_labels = input_parameters
    method, best_score, test_score = tree_classification(training_set_features, testing_set_features, 
                                                         training_set_labels, testing_set_labels)
    assert method == 'decision_tree', "Method is not correct"
    assert 0.0 <= best_score <= 1.0, "Best Score is not within the correct range"
    assert 0.0 <= test_score <= 1.0, "Test Score is not within the correct range"