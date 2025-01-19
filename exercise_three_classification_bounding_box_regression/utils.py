import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def calculate_loss(class_outputs, reg_outputs, targets, device, num_classes):
    class_to_idx = {'dog': 0, 'cat': 1}
    total_loss = 0

    for i in range(4):  # Loop through each patch
        class_targets_tensor = F.one_hot(targets[i]['label'], num_classes=num_classes).to(device).float()
        bbox_targets_tensor = targets[i]['bbox'].to(device)

        # Classification Loss
        if len(class_targets_tensor) > 0:
            classification_loss = nn.MSELoss()(class_outputs[i], class_targets_tensor)
        else:
            classification_loss = torch.tensor(0.0, device=device)

        # Regression Loss
        if len(bbox_targets_tensor) > 0:
            regression_loss = nn.MSELoss()(reg_outputs[i], bbox_targets_tensor)
        else:
            regression_loss = torch.tensor(0.0, device=device)

        total_loss += classification_loss + regression_loss

    return total_loss


def evaluate_model(model, data_loader, device, num_classes, class_to_idx):
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for patches, patch_annotations in data_loader:
            patches = patches.to(device)

            class_outputs, reg_outputs = model(patches)

            total_loss = calculate_loss(class_outputs, reg_outputs, patch_annotations, device, num_classes)

            # Calculate accuracy
            for i in range(4):
                class_predictions = class_outputs[i].argmax(dim=1)
                class_targets = patch_annotations[i]['label']
                correct_val += (class_predictions == class_targets.to(device)).sum().item()

                total_val += len(class_targets)

            running_loss += total_loss.item()

    val_loss = running_loss / len(data_loader)
    val_accuracy = correct_val / total_val if total_val > 0 else 0
    return val_loss, val_accuracy


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, num_classes, class_to_idx):
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for patches, patch_annotations in tqdm.tqdm(train_loader):
            patches = patches.to(device)

            optimizer.zero_grad()
            class_outputs, reg_outputs = model(patches)

            total_loss = calculate_loss(class_outputs, reg_outputs, patch_annotations, device, num_classes)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, device, num_classes, class_to_idx)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    return train_losses, val_losses, train_accuracies, val_accuracies


