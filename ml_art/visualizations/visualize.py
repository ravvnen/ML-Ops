import matplotlib.pyplot as plt
import os
import pandas as pd
import wandb
import glob
import numpy as np
import torch

from tqdm import tqdm


def plot_model_performance(log_path, model_name):
    # Meant for training/training-testing, for testing accuracy is a sufficient metric
    # Load Model Performances

    df = pd.read_csv(glob.glob(os.path.join(log_path, "*.csv"))[0])

    # To Avoid Any Unbound variables
    training_losses = []
    testing_losses = []
    training_accuracies = []
    testing_accuracies = []

    if "Training Loss" in df.columns:
        training_losses = df["Training Loss"].to_list()

    if "Training Accuracy" in df.columns:
        training_accuracies = df["Training Accuracy"].to_list()

    if "Testing Loss" in df.columns:
        testing_losses = df["Testing Loss"].to_list()

    if "Testing Accuracy" in df.columns:
        testing_accuracies = df["Testing Accuracy"].to_list()

    # Plot training and testing loss
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)

    if training_losses:
        plt.plot(
            range(1, len(training_losses) + 1),
            training_losses,
            label="Training Loss",
        )

    if testing_losses:
        plt.plot(
            range(1, len(testing_losses) + 1),
            testing_losses,
            label="Testing Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: Training and Testing Loss over Epochs")
    plt.legend()
    fig.savefig(
        os.path.join(log_path, f"{model_name}_loss_plot.png"),
        dpi=200,
        bbox_inches="tight",
    )

    if wandb.run:
        wandb.log({"Loss": fig})

    # Plot training and testing accuracy
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)

    if training_accuracies:
        plt.plot(
            range(1, len(training_accuracies) + 1),
            training_accuracies,
            label="Training Accuracy",
        )

    if testing_accuracies:
        plt.plot(
            range(1, len(testing_accuracies) + 1),
            testing_accuracies,
            label="Testing Accuracy",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name}: Training and Testing Accuracy over Epochs")
    plt.legend()
    fig.savefig(
        os.path.join(log_path, f"{model_name}_accuracy_plot.png"),
        dpi=200,
        bbox_inches="tight",
    )

    if wandb.run:
        wandb.log({"Accuracy": fig})


def view_scores(scores, cfg):
    # Create a new figure
    fig = plt.figure()

    # Add a subplot to the figure
    ax = fig.add_subplot(1, 1, 1)  # 1 row, 1 column, first subplot

    ax.barh(np.arange(len(cfg.dataset.styles)), scores)
    ax.set_aspect(1.1)
    ax.set_yticks(np.arange(len(cfg.dataset.styles)))
    ax.set_yticklabels(cfg.dataset.styles, size="small")
    ax.tick_params(axis="y", which="major", labelsize=16)
    ax.set_xlim(0, 1)
    plt.close()

    return fig


def wandb_table(data_loader, preds, config, logger):
    tbl = wandb.Table(
        columns=["image", "scores", "prediction", "ground_truth"]
    )

    data_loader = tqdm(data_loader, desc="Creating W&B Table", ascii=True)

    for imgs, targets in data_loader:
        targets_to_style = [
            config.dataset.styles[target.item()] for target in targets
        ]
        preds_to_style = [
            config.dataset.styles[pred.item()]
            for pred in torch.argmax(preds, dim=1)
        ]
        [
            tbl.add_data(
                wandb.Image(img),
                wandb.Image(view_scores(score, config)),
                pred,
                target,
            )
            for img, score, pred, target in zip(
                imgs, preds.cpu(), preds_to_style, targets_to_style
            )
        ]

        # Figures created in view_scores are closed for memory
        plt.close("all")

        data_loader.set_postfix()
        logger.info(str(data_loader))

    return tbl
