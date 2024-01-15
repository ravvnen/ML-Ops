import os
import torch
import pytest
import logging
import tempfile
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch, MagicMock
from ml_art.train_model import train


# Fixture for a dummy dataloader
@pytest.fixture
def dummy_dataloader():
    # Create a small dataset with random data
    # Testing for binary classification targets
    inputs = torch.randn(10, 3, 512, 512)
    targets = torch.randint(0, 2, (10,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=2)


# Simple mock model
class MockModel(torch.nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = torch.nn.Linear(512 * 512 * 3, 2)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


# Test the train function
@patch("hydra.core.hydra_config.HydraConfig.get")
@patch("torch.optim.Adam")
def test_train(mock_optimizer, mock_hydra_get, dummy_dataloader):
    model = MockModel()
    mock_optimizer_instance = MagicMock()
    mock_optimizer.return_value = mock_optimizer_instance

    with tempfile.TemporaryDirectory() as temp_output_dir:
        # Mock configuration
        cfg = MagicMock()
        cfg.runtime.output_dir = temp_output_dir
        cfg.hyperparameters = MagicMock(lr=0.001, num_epochs=1)
        cfg.model = "model"

        mock_hydra_get.return_value = cfg

        # Mock logger
        logger = logging.getLogger("test")

        # Call the train function
        train(model, dummy_dataloader, cfg, logger)

        # Assert that the optimizer's step function was called
        mock_optimizer_instance.step.assert_called()

        # Check if the model file was saved in the temporary directory
        model_file = os.path.join(temp_output_dir, cfg.model + ".pth")
        assert os.path.exists(model_file), "Model file not saved"
