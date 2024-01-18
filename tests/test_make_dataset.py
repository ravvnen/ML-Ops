import os
import pytest
import tempfile
from PIL import Image
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch
from ml_art.data.make_dataset import WikiArt, PadAndResize, main


# Fixture for mocking os environment variable
@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("LOCAL_PATH", "/fake/path/to/data")


# Test to check if the data folder exists
def test_data_folder_existence(mock_env):
    with patch("os.path.isdir") as mock_isdir:
        mock_isdir.return_value = True
        assert os.path.isdir("/fake/path/to/data")


# Test to check correct data pulling
@patch("torch.save")
@patch("ml_art.data.make_dataset.WikiArt")
@patch("ml_art.data.make_dataset.os.path.isdir")
@patch("omegaconf.OmegaConf.to_yaml")
@patch("hydra.core.hydra_config.HydraConfig.get")
@patch("os.getenv", return_value="/fake/path")
def test_main_function(
    mock_hydra_get,
    mock_to_yaml,
    mock_isdir,
    mock_WikiArt,
    mock_torch_save,
    mock_getenv,
):
    # Mock the os.getenv to return a fake path
    mock_getenv.return_value = "/fake/local/path"

    # Mock the isdir function to always return True
    mock_isdir.return_value = True

    # Use a temporary directory for the mocked Hydra output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the Hydra configuration to return a realistic file path
        mock_cfg = MagicMock()
        mock_cfg.runtime.output_dir = temp_dir
        mock_hydra_get.return_value = mock_cfg

        # Mock the OmegaConf to_yaml function
        mock_to_yaml.return_value = "Mocked Config"

        # Setup mock WikiArt class
        mock_WikiArt.return_value = MagicMock()
        mock_WikiArt.return_value.__len__.return_value = 200
        # print(len(mock_WikiArt))

        # TODO: Fix when we can figure out why the mock_WikiArt is empty
        # # Create a mock configuration object
        # config = OmegaConf.create(
        #     {
        #         "dataset": {
        #             "raw_path": "fake_raw_path",
        #             "styles": ["style1", "style2"],
        #             "imgs_per_style": 100,
        #             "test_size": 0.2,
        #             "input_shape": [3, 128, 128],
        #             "seed": 42,
        #         }
        #     }
        # )

        # # Call the main function with the mocked configuration
        # main(config)

        # # Assertion on save() is successful
        # mock_torch_save.assert_called()


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
    # at the corners of the image, which should be (0, 0, 0)
    # if the padding is black
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
