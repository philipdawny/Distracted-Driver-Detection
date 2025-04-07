import torch
from tqdm import tqdm
from utils import plot_metrics

def train_model(model, train_loader, val_loader, device, config):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        correct_train_preds = 0
        total_train_samples = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train_preds += predicted.eq(labels).sum().item()
            total_train_samples += labels.size(0)

        train_accuracy = 100 * correct_train_preds / total_train_samples
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        scheduler.step(val_accuracy)

        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    # Plot metrics after training is complete
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, config["num_epochs"])

def evaluate_model(model, loader, device):
    """
    Evaluates the model on a given dataset (validation or test).
    
    Returns:
        - Average loss over the dataset.
        - Accuracy (%).
    """
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
            total_samples += labels.size(0)

    accuracy = 100 * correct_predictions / total_samples
    return total_loss / len(loader), accuracy
