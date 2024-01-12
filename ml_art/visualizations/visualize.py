import matplotlib.pyplot as plt
import os
import pandas as pd
import wandb
import glob


def plot_model_performance(log_path, model_name):
    # Meant for training/training-testing, for testing accuracy is a sufficient metric
    # Load Model Performances

    df = pd.read_csv(glob.glob(os.path.join(log_path, "*.csv"))[0])

    try:
        training_losses = df["Training Loss"].to_list()
        training_accuracies = df["Training Accuracy"].to_list()
    except Exception as e:
        print(e)

    try:
        testing_losses = df["Testing Loss"].to_list()
        testing_accuracies = df["Testing Accuracy"].to_list()
    except Exception as e:
        print(e)

    # Plot training and testing loss
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)

    try:
        plt.plot(
            range(1, len(training_losses) + 1),
            training_losses,
            label="Training Loss",
        )
    except Exception as e:
        print(e)

    try:
        plt.plot(
            range(1, len(training_losses) + 1),
            testing_losses,
            label="Testing Loss",
        )
    except Exception as e:
        print(e)

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
    try:
        plt.plot(
            range(1, len(training_losses) + 1),
            training_accuracies,
            label="Training Accuracy",
        )
    except Exception as e:
        print(e)
    try:
        plt.plot(
            range(1, len(training_losses) + 1),
            testing_accuracies,
            label="Testing Accuracy",
        )
    except Exception as e:
        print(e)

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
