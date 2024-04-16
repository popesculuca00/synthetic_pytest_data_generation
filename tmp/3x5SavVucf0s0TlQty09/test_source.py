# test_source.py
import source  # Importing the source file
import pytest  # Pytest framework

class TestSource:
    def test_client_color_to_rgb(self):
        # Testing for value < 0
        assert source.client_color_to_rgb(-1) == 0, "Failed when input is -1"
        # Testing for value > 215
        assert source.client_color_to_rgb(216) == 0, "Failed when input is 216"
        # Testing for value in range [0, 215]
        assert source.client_color_to_rgb(120) != 0, "Failed when input is in range [0, 215]"