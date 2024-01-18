import torch
import pytest
from omegaconf import DictConfig
from unittest.mock import MagicMock, patch
from ml_art.predict_model import predict


# Fixture for a mock DataLoader
@pytest.fixture
def mock_dataloader():
    # Create a mock DataLoader that yields a batch of images and labels
    mock_loader = MagicMock()
    mock_loader.__iter__.return_value = iter(
        [(torch.randn(2, 3, 128, 128), torch.randint(0, 2, (2,)))]
    )
    return mock_loader


# Test for the predict function
@patch("ml_art.predict_model.torch.load")  # Mock torch.load
@patch("ml_art.predict_model.timm.create_model")  # Mock timm.create_model
@patch("os.getenv")  # Mock os.getenv to provide a value for LOCAL_PATH
def test_predict(
    mock_getenv, mock_create_model, mock_torch_load, mock_dataloader
):
    # Mock the return value of os.getenv for LOCAL_PATH
    mock_getenv.return_value = "/mock/path/to/weights"

    # Setup mock model to return a tensor
    mock_model = MagicMock()
    mock_model.return_value = torch.randn(2, 2)  # Mock output tensor
    mock_create_model.return_value = mock_model

    # Mock the state_dict loading
    mock_torch_load.return_value = MagicMock()

    # Mock configuration
    cfg = DictConfig({"model": "test_model", "weights": "path/to/weights"})

    # Mock logger
    logger = MagicMock()

    # Call the predict function
    result = predict(mock_model, mock_dataloader, cfg, logger, None)

    # Assertions
    assert isinstance(
        result, torch.Tensor
    ), "Prediction result should be a torch.Tensor"

    # Check that logger was called
    logger.info.assert_called()
