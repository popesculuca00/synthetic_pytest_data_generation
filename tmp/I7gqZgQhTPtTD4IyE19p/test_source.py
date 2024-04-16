import torch
import source  # assuming source.py is your original file

def test_trilinear_composition():
    # Testing values
    h_s = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    h_t = torch.tensor([[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]])
    x = torch.tensor([[[25, 26, 27], [28, 29, 30]], [[31, 32, 33], [34, 35, 36]]])

    # Running the method
    result = source.trilinear_composition(h_s, h_t, x)

    # Assertion
    assert torch.allclose(result, torch.tensor([[[51, 66, 81], [139, 164, 189]], [[204, 238, 262], [337, 362, 387]]]))