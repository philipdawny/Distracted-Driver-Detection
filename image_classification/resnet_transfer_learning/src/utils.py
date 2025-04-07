import matplotlib.pyplot as plt
import os

def plot_and_save_metric(metric_values, metric_name, ylabel, title, filename):


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(metric_values) + 1), metric_values, marker='o', label=metric_name)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=15)
    plt.legend()
    plt.grid(True)

    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the plot
    plt.savefig(filename)
    print(f"Saved {metric_name} plot to {filename}")
    plt.close()


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs):
    
    
    output_dir = "plots/"

    
    plot_and_save_metric(
        train_losses,
        "Training Loss",
        ylabel="Loss",
        title="Training Loss Over Epochs",
        filename=os.path.join(output_dir, "training_loss.png")
    )

    
    plot_and_save_metric(
        val_losses,
        "Validation Loss",
        ylabel="Loss",
        title="Validation Loss Over Epochs",
        filename=os.path.join(output_dir, "validation_loss.png")
    )

    
    plot_and_save_metric(
        train_accuracies,
        "Training Accuracy",
        ylabel="Accuracy (%)",
        title="Training Accuracy Over Epochs",
        filename=os.path.join(output_dir, "training_accuracy.png")
    )

    
    plot_and_save_metric(
        val_accuracies,
        "Validation Accuracy",
        ylabel="Accuracy (%)",
        title="Validation Accuracy Over Epochs",
        filename=os.path.join(output_dir, "validation_accuracy.png")
    )



def display_test_results(images, predictions, true_labels=None, class_labels=None):
    """
    Displays a grid of test images with their predicted (and optionally true) labels.

    Args:
        images: Batch of images (tensor).
        predictions: Predicted labels.
        true_labels: True labels (optional).
        class_labels: List of class names corresponding to label indices.
    """
    num_images = len(images)
    rows = int(np.ceil(num_images / 5))
    cols = min(5, num_images)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    
    for idx in range(num_images):
        ax = axes[idx // cols][idx % cols] if rows > 1 else axes[idx % cols]
        
        # Unnormalize image
        img = images[idx].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        
        # Set title with prediction and optionally true label
        pred_label = class_labels[predictions[idx]] if class_labels else predictions[idx]
        
        if true_labels is not None:
            true_label = class_labels[true_labels[idx]] if class_labels else true_labels[idx]
            color = "green" if predictions[idx] == true_labels[idx] else "red"
            ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        else:
            ax.set_title(f"Pred: {pred_label}")
        
        ax.axis("off")
    
    # Hide unused subplots
    for idx in range(num_images, rows * cols):
        fig.delaxes(axes.flatten()[idx])

    plt.tight_layout()
    plt.show()