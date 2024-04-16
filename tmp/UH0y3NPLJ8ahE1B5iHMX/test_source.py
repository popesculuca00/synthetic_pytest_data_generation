import torch
import torch.testing as t
import unittest
import numpy as np

# Import the source.py file
from source import invert_pose

class TestInvertPose(unittest.TestCase):

    def test_invert_pose(self):
        # create random transformation matrix
        T01 = torch.randn(4, 4, 4, dtype=torch.float32, device='cuda')
        # generate the expected output
        expected_output = invert_pose(T01)
        # check if the output is as expected
        t.assertclose(expected_output, invert_pose(T01))

if __name__ == '__main__':
    unittest.main()