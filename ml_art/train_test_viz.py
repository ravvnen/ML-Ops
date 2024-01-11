import os
import torch
import hydra
import timm
import random
import logging
import omegaconf

import torch.optim as optim
import torch.nn as nn
import pandas as pd

from ml_art.data.data import wiki_art
from tqdm import tqdm
from ml_art.models.model import ArtCNN
from typing import Union
from ml_art.visualizations.visualize import plot_model_performance

# Needed For Loading a Dataset created using WikiArt & pad_resize in make_dataset.py
from ml_art.data.make_dataset import WikiArt, PadAndResize


def train_test_viz(
    model: Union[
        torch.nn.Module,
        timm.models.resnet.ResNet,
        timm.models.efficientnet.EfficientNet,
    ],
    data_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    cfg: omegaconf.dictconfig.DictConfig,
    logger: logging.Logger,
) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
        cfg: configuration file
        logger: logger object

    Returns
        None: Saves weights & KPIs in Hydra Default Log Dir

    """

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hydra Configuration For Model Setup
    model_cfg = cfg.model
    hp = cfg.hyperparameters

    # Start Training
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    training_losses, training_accuracies = [], []
    testing_losses, testing_accuracies = [], []

    # Typical Training Snippet
    for epoch in tqdm(range(hp.epochs), desc="Epochs", ascii=True):
        logger.info(f"Train Epoch: {epoch + 1}")

        running_loss = 0.0
        correct = 0
        total = 0

        data_loader = tqdm(
            data_loader, desc="Training", unit="batch", ascii=True
        )

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # # Update the progress bar
            data_loader.set_postfix(
                loss=(running_loss / total), accuracy=(100 * correct / total)
            )

            logger.info(str(data_loader))

        # Store KPIs
        avg_train_loss = running_loss / len(data_loader)
        train_accuracy = 100 * correct / total

        training_losses.append(avg_train_loss)
        training_accuracies.append(train_accuracy)

        logger.info(f"Test Epoch: {epoch + 1}")
        # Model Already on Device
        model.eval()

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

        testing_losses.append(avg_test_loss)
        testing_accuracies.append(test_accuracy)

    # Log KPIs & weights
    hydra_log_dir = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )

    torch.save(
        model.state_dict(), os.path.join(hydra_log_dir, model_cfg + ".pth")
    )
    logger.info(f"Saved Weights to {hydra_log_dir}")

    df = pd.DataFrame(
        {
            "Training Loss": training_losses,
            "Training Accuracy": training_accuracies,
            "Testing Loss": testing_losses,
            "Testing Accuracy": testing_accuracies,
        }
    )
    df.to_csv(os.path.join(hydra_log_dir, "train_test_log.csv"), index=False)
    logger.info(f"Saved loss & accuracy to {hydra_log_dir}")

    plot_model_performance(df, cfg.model, hydra_log_dir)


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

    # Get Data Loader
    train_loader, test_loader = wiki_art(config)

    # Choose model from Hydra config file in ml_art/config
    if model_cfg != "CNN":
        # Try models in timm
        try:
            model = timm.create_model(
                model_cfg, num_classes=len(data_cfg.styles), pretrained=False
            )
        except Exception as e:
            print(f"Error: {e}")
            print("Model unknown")

    else:
        # Our custom model
        model = ArtCNN(config)

    train_test_viz(model, train_loader, test_loader, config, logger)


if __name__ == "__main__":
    main()
