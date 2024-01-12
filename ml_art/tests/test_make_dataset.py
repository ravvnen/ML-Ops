import os
import pytest
from PIL import Image
from ml_art.data.make_dataset import WikiArt, PadAndResize

# from unittest.mock import patch


# Mock os.listdir to return a controlled list of filenames
@pytest.fixture
def mock_listdir(monkeypatch):
    def mock_return(path):
        return ["image1.jpg", "image2.jpg", "image3.jpg"]

    monkeypatch.setattr(os, "listdir", mock_return)


# Mock os.path.join to just concatenate strings
@pytest.fixture
def mock_path_join(monkeypatch):
    def mock_return(*args):
        return "/".join(args)

    monkeypatch.setattr(os.path, "join", mock_return)


# Mock os.path.isdir to always return True
@pytest.fixture
def mock_isdir(monkeypatch):
    monkeypatch.setattr(os.path, "isdir", lambda x: True)


def test_get_random_image_by_style(mock_listdir, mock_path_join, mock_isdir):
    root_dir = "some/testing/path"
    styles = ["Baroque"]
    art_dataset = WikiArt(root_dir, styles)

    # Act
    random_image_path = art_dataset.get_random_image_by_style("Baroque")

    # Assert
    assert random_image_path.startswith(root_dir)
    assert random_image_path.endswith(".jpg")
    assert "Baroque" in random_image_path


# Test that PadAndResize correctly pads and resizes an image
def test_pad_and_resize():
    # Create a small dummy image for testing
    input_image = Image.new("RGB", (800, 600), color="red")

    # Target size for padding and resizing
    target_size = (1024, 1024)

    # Create an instance of PadAndResize
    pad_and_resize = PadAndResize(target_size)

    # Apply the transformation
    output_image = pad_and_resize(input_image)

    # Check that the output image has the expected dimensions
    assert output_image.size == target_size

    # Check that padding was applied correctly by examining the pixel values
    # at the corners of the image, which should be (0, 0, 0) if the padding is black
    corners = [
        (0, 0),  # Top-left corner
        (0, target_size[1] - 1),  # Bottom-left corner
        (target_size[0] - 1, 0),  # Top-right corner
        (target_size[0] - 1, target_size[1] - 1),  # Bottom-right corner
    ]
    for x, y in corners:
        assert output_image.getpixel((x, y)) == (
            0,
            0,
            0,
        ), f"Pixel at {(x, y)} is not padded correctly."
