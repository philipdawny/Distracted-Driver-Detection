import json
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from .utils import display_test_results

def plot_training_metrics(log_history, output_dir="plots"):
    
    metrics = {"loss": [], "eval_loss": [], "eval_accuracy": []}

    for entry in log_history:
        for key in metrics:
            if key in entry:
                metrics[key].append(entry[key])

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(metrics["loss"], label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()

    # Plot validation loss
    plt.figure(figsize=(10, 4))
    plt.plot(metrics["eval_loss"], label="Validation Loss", color="orange")
    plt.title("Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "validation_loss.png"))
    plt.close()

    # Plot validation accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(metrics["eval_accuracy"], label="Validation Accuracy", color="green")
    plt.title("Validation Accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "validation_accuracy.png"))
    plt.close()

    print(f"Plots saved to {output_dir}")
