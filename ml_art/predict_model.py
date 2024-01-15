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
import time

import torch.nn as nn

from ml_art.data.data import wiki_art
from tqdm import tqdm
from ml_art.models.model import ArtCNN
from typing import Union
from hydra.core.hydra_config import HydraConfig

# Needed For Loading a Dataset created using WikiArt & pad_resize
from ml_art.data.make_dataset import WikiArt, PadAndResize

from ml_art.visualizations.visualize import wandb_table


def predict(
    model: Union[
        torch.nn.Module,
        timm.models.resnet.ResNet,
        timm.models.efficientnet.EfficientNet,
    ],
    test_loader: torch.utils.data.DataLoader,
    cfg: omegaconf.dictconfig.DictConfig,
    logger: logging.Logger,
    profiler: Union[torch.profiler.profile, None],
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
    root_dir = os.getenv("LOCAL_PATH")
    if root_dir is not None:
        state_dict = torch.load(
            os.path.join(root_dir, cfg.weights, model_cfg + ".pth")
        )
        model.load_state_dict(state_dict)
    else:
        raise ValueError("LOCAL_PATH not set")

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

    total_outputs = []

    with torch.no_grad():
        for images, labels in test_loader:
            # Very Important For Profiling
            if profiler is not None:
                profiler.step()
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            total_outputs.append(outputs)

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

    tmp = torch.cat(total_outputs)

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
    # Get Data loader
    _, test_loader = wiki_art(config)
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
            pred_scores = predict(model, test_loader, config, logger, prof)

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
        pred_scores = predict(model, test_loader, config, logger, None)

    pred_probs = torch.nn.functional.softmax(pred_scores, dim=1)
    table = wandb_table(test_loader, pred_probs, config, logger)
    wandb.log({"WikiArt Classification": table})


if __name__ == "__main__":
    main()
