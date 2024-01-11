import os
import torch
import hydra
import timm
import random
import logging
import omegaconf

import torch.nn as nn

from ml_art.data.data import wiki_art
from tqdm import tqdm
from ml_art.models.model import ArtCNN
from typing import Union  # <class 'timm.models.resnet.ResNet'>

# Needed For Loading a Dataset created using WikiArt & pad_resize
from ml_art.data.make_dataset import WikiArt, pad_and_resize


def predict(
    model: Union[
        torch.nn.Module,
        timm.models.resnet.ResNet,
        timm.models.efficientnet.EfficientNet,
    ],
    test_loader: torch.utils.data.DataLoader,
    cfg: omegaconf.dictconfig.DictConfig,
    logger: logging.Logger,
) -> torch.Tensor:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hydra Configuration For Model Setup
    model_cfg = cfg.model

    # Load Weights
    state_dict = torch.load(os.path.join(cfg.weights, model_cfg + ".pth"))
    model.load_state_dict(state_dict)

    # Predict
    model.eval()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Typical Prediction Snippet
    test_loss = 0.0
    correct = 0
    total = 0

    test_loader = tqdm(
        test_loader, desc="Evaluating", unit="batch", ascii=True
    )

    total_output = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            total_output.append(outputs)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update the progress bar
            test_loader.set_postfix(
                loss=(test_loss / total), accuracy=(100 * correct / total)
            )

            logger.info(str(test_loader))

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    for i, output in enumerate(total_output):
        if i == 0:
            tmp = total_output[i]
        else:
            tmp = torch.cat(total_output)

    logger.info("Model Output: %s", str(output))
    logger.info("Output Shape: %s", str(output.shape))
    logger.info("Average Test Loss: %s", str(avg_test_loss))
    logger.info("Test Accuracy: %s", str(test_accuracy))

    return tmp


@hydra.main(
    config_path="config", config_name="config.yaml", version_base="1.1"
)
def main(config):
    # Init Logger - Hydra sets log dirs to outputs/ by default
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Hydra Configuration For Model Setup
    data_cfg = config.dataset
    model_cfg = config.model

    # Log cfg
    logger.info(f"configuration: \n {omegaconf.OmegaConf.to_yaml(config)}")

    # Ensure Reproducibility
    torch.manual_seed(data_cfg.seed)
    random.seed(data_cfg.seed)

    _, test_loader = wiki_art(config)

    if model_cfg != "CNN":
        try:
            model = timm.create_model(
                model_cfg, num_classes=len(data_cfg.styles), pretrained=False
            )
        except Exception as e:
            logger.info(f"Error: {e}")
            logger.info("Model unknown")

    else:
        model = ArtCNN(config)

    predict(model, test_loader, config, logger)


if __name__ == "__main__":
    main()
