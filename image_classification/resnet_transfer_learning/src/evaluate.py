import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def test_model(model, test_loader, device, loss_fn=None, labeled=False):
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for inputs in test_loader:
            if labeled:
                inputs, labels = inputs
                inputs, labels = inputs.to(device), labels.to(device)
            else:
                inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())

            if labeled:
                all_labels.extend(labels.cpu().numpy())
                if loss_fn:
                    total_loss += loss_fn(outputs, labels).item()

    if labeled:
        accuracy = accuracy_score(all_labels, all_predictions) * 100
        avg_loss = total_loss / len(test_loader) if loss_fn else None
        return accuracy, avg_loss, all_predictions, all_labels
    
    return all_predictions