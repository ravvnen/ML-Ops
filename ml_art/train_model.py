import os
import torch
import hydra
import timm
import random
import logging
import omegaconf
import wandb
import glob
import warnings

import torch.optim as optim
import torch.nn as nn
import pandas as pd
import yaml

from ml_art.data.data import wiki_art
from tqdm import tqdm
from ml_art.models.model import ArtCNN
from typing import Union
from hydra.core.hydra_config import HydraConfig

# Needed For Loading a Dataset created using WikiArt & pad_resize in make_dataset.py
from ml_art.data.make_dataset import WikiArt, PadAndResize
from ml_art.visualizations.visualize import plot_model_performance


def config_weight_path_edit(file_path, new_value):
    # Load YAML data from the file
    with open(file_path, "r") as file:
        yaml_data = yaml.safe_load(file)

    # Edit the specified key in the YAML data
    yaml_data["weights"] = new_value

    # Write the updated YAML data back to the file
    with open(file_path, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)


def train(
    model: Union[
        torch.nn.Module,
        timm.models.resnet.ResNet,
        timm.models.efficientnet.EfficientNet,
    ],
    data_loader: torch.utils.data.DataLoader,
    cfg: omegaconf.dictconfig.DictConfig,
    logger: logging.Logger,
    profiler: Union[torch.profiler.profile, None],
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

    # Typical Training Snippet
    for epoch in tqdm(range(hp.epochs), desc="Epochs", ascii=True):
        logger.info(f"Epoch: {epoch + 1}")

        running_loss = 0.0
        correct = 0
        total = 0

        data_loader = tqdm(
            data_loader, desc="Training", unit="batch", ascii=True
        )

        for images, labels in data_loader:
            # Very Important For Profiling
            if profiler is not None:
                profiler.step()

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
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = 100 * correct / total

        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_acc)

    # Log KPIs & weights
    hydra_log_dir = HydraConfig.get().runtime.output_dir

    # Save Weights
    torch.save(
        model.state_dict(), os.path.join(hydra_log_dir, model_cfg + ".pth")
    )
    logger.info(f"Saved Weights to {hydra_log_dir}")

    df = pd.DataFrame(
        {
            "Training Loss": training_losses,
            "Training Accuracy": training_accuracies,
        }
    )
    df.to_csv(os.path.join(hydra_log_dir, "training_log.csv"), index=False)
    logger.info(f"Saved training loss & accuracy to {hydra_log_dir}")


@hydra.main(
    config_path="config", config_name="config.yaml", version_base="1.1"
)
def main(config):
    # Init Logger - Hydra sets log dirs to outputs/ by default
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    hydra_log_dir = HydraConfig.get().runtime.output_dir

    # ML Experiment Tracking Platform (Requires W&B Account -> Will ask for API Key)
    config_dict = omegaconf.OmegaConf.to_container(config, resolve=True)
    if isinstance(config_dict, dict):
        wandb.init(
            project="ml-art",
            config=config_dict,
            sync_tensorboard=True,
        )
        # Suppress UserWarnings from plotly
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="plotly"
        )
    else:
        raise ValueError("Config must be a dictionary.")

    # Hydra Configuration For Model Setup
    data_cfg = config.dataset
    model_cfg = config.model

    # Log cfg
    logger.info(f"configuration: \n {omegaconf.OmegaConf.to_yaml(config)}")

    # Ensure Reproducibility
    torch.manual_seed(data_cfg.seed)
    random.seed(data_cfg.seed)

    # Get Data Loader
    train_loader, _ = wiki_art(config)

    # Choose model from Hydra config file in ml_art/config
    if model_cfg != "CNN":
        # Try models in timm
        try:
            model = timm.create_model(
                model_cfg, num_classes=len(data_cfg.styles), pretrained=False
            )
        except Exception as e:
            logger.info(f"Error: {e}")
            logger.info("Model unknown")
            raise ValueError("Model unknown")

    else:
        # Our custom model
        model = ArtCNN(config)

    # Enable Profiler for examining bottlenecks
    if config.profile is True:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                hydra_log_dir
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            train(model, train_loader, config, logger, prof)

        if torch.cuda.is_available():
            logger.info(
                prof.key_averages().table(
                    sort_by="cpu_time_total", row_limit=10
                )
            )
            logger.info(
                prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=10
                )
            )
        else:
            logger.info(
                prof.key_averages().table(
                    sort_by="cpu_time_total", row_limit=10
                )
            )

        # Save Trace to W&B (Requieres administator rights)
        files_with_pattern = glob.glob(
            os.path.join(hydra_log_dir, "*.pt.trace.json")
        )
        if len(files_with_pattern) > 0:
            trace_path = files_with_pattern[0]
            wandb.save(trace_path)
        else:
            logger.info("No trace file found")

    # Run without profiler
    else:
        train(model, train_loader, config, logger, None)

    # Visualize KPIs
    plot_model_performance(hydra_log_dir, config.model)

    # Set weights in config file (Automatically as compared to manually)
    root_dir = os.getenv("LOCAL_PATH")
    relative_path = os.path.relpath(hydra_log_dir, root_dir)
    if root_dir is not None:
        config_file_path = os.path.join(
            root_dir, "ml_art/config", "config.yaml"
        )
        config_weight_path_edit(config_file_path, relative_path)
        logger.info(f"Set weights in config file to:  {relative_path}")


if __name__ == "__main__":
    main()
