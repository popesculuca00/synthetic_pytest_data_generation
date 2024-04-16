import pytest
import sys
sys.path.append(".")  # To import source.py from the same directory
from source import augmented_image_transforms  # Import the function from source.py
import torchvision.transforms as transforms  # Required import for the function

# We will test the function with some specific values to ensure it works as expected

def test_augmented_image_transforms():
    # Declare instance of augmented_image_transforms
    augmented_image_transforms_instance = augmented_image_transforms(d=10, t=0.4, s=0.3, sh=10, ph=0.6, pv=0.8, resample=3)

    # Check that the returned object is an instance of transforms.Compose
    assert isinstance(augmented_image_transforms_instance, transforms.Compose)

if __name__ == "__main__":
    pytest.main()