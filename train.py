import torch
import torch.nn as nn
import os
from model import AlexNet
from data_loader import get_train_valid_loader, get_test_loader


def setup_directories(config):
    """Create necessary directories for the training process"""
    # Setup data directory
    if not os.path.exists(config["data_dir"]):
        print(f"Creating data directory: {config['data_dir']}")
        os.makedirs(config["data_dir"], exist_ok=True)
    else:
        print(f"Data directory already exists: {config['data_dir']}")
    
    # Setup checkpoint directory
    if not os.path.exists(config["checkpoint_dir"]):
        print(f"Creating checkpoint directory: {config['checkpoint_dir']}")
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
    else:
        print(f"Checkpoint directory already exists: {config['checkpoint_dir']}")


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup directories
    setup_directories(config)
    
    print(f"Loading CIFAR-10 dataset. If not present, it will be downloaded to {config['data_dir']}")
    train_loader, valid_loader = get_train_valid_loader(data_dir=config["data_dir"], batch_size=config["batch_size"])
    test_loader = get_test_loader(data_dir=config["data_dir"], batch_size=config["batch_size"])
    print("Dataset loaded successfully!")

    model = AlexNet(num_classes=config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        momentum=config["momentum"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=2
    )

    total_step = len(train_loader)

    best_accuracy = 0.0
    best_model_path = os.path.join(config["checkpoint_dir"], "best_model.pth")
    patience = config["patience"]
    patience_counter = 0

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, config["num_epochs"], i + 1, total_step, loss.item()))
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], Average Loss: {avg_epoch_loss:.4f}")

        current_accuracy = validate(model, valid_loader, device, mode="validation")
        scheduler.step(current_accuracy)
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': current_accuracy,
                'loss': avg_epoch_loss,
            }, best_model_path)
            print(f"Model checkpoint saved at {best_model_path} with accuracy: {best_accuracy:.2f}%")
            
            model_only_path = os.path.join(config["checkpoint_dir"], "best_model_state.pth")
            torch.save(model.state_dict(), model_only_path)
            print(f"Best model state saved at {model_only_path}")
        else:
            patience_counter += 1
            print(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                checkpoint = torch.load(best_model_path, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['accuracy']:.2f}%")
                break

    # Test with best model
    print("\nEvaluating best model on test set...")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Using best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['accuracy']:.2f}%")
    
    test_accuracy = validate(model, test_loader, device, mode="test")
    
    return model, best_accuracy, test_accuracy


def validate(model, data_loader, device, mode="validation"):
    """Evaluate model performance on validation or test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the {total} {mode} images: {accuracy:.2f}%")
    
    return accuracy


if __name__ == "__main__":
    config = {
        "num_classes": 10,
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 0.005,
        "weight_decay": 0.005,
        "momentum": 0.9,
        "data_dir": "./data",
        "checkpoint_dir": "./checkpoints",
        "patience": 5,
    }

    model, best_val_accuracy, test_accuracy = train(config)
    print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%")