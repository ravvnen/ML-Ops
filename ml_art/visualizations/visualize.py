import matplotlib.pyplot as plt
import os


def plot_model_performance(df,model_name,log_path):


    # Load Model Performances
    training_losses = df["Training Loss"].to_list()
    training_accuracies = df["Training Accuracy"].to_list()

    testing_losses = df["Testing Loss"].to_list()
    testing_accuracies = df["Testing Accuracy"].to_list()


    # Plot training and testing loss
    fig,ax=plt.subplots(figsize=(10, 4), dpi=200)
    plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
    plt.plot(range(1, len(training_losses) + 1), testing_losses, label="Testing Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: Training and Testing Loss over Epochs")
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(log_path,f"{model_name}_loss_plot.png"), dpi=200, bbox_inches="tight")

    # Plot training and testing accuracy
    fig,ax=plt.subplots(figsize=(10, 4), dpi=200)
    plt.plot(range(1, len(training_losses) + 1), training_accuracies, label="Training Accuracy")
    plt.plot(range(1, len(training_losses) + 1), testing_accuracies, label="Testing Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{model_name}: Training and Testing Accuracy over Epochs")
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(log_path,f"{model_name}_accuracy_plot.png"), dpi=200, bbox_inches="tight")
