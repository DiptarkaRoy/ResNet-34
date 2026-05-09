import gc
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
# from model import resnet34
# from dataset import get_dataloaders
from src import resnet34, get_dataloaders

def model_setup(num_classes=10):
    device=torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f'Using device: {device}')

    # Instantiate the ResNet-34 model
    model=resnet34(num_classes=num_classes)
    print(model)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
        
    model.to(device)

    # Define the loss function - CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer with weight decay and momentum for better convergence
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # Use a Cosine scheduler for a smooth decay over the full 50 epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    return model, criterion, optimizer, scheduler, device

def train(model, criterion, optimizer, scheduler, image_size, train_loader, test_loader, device, num_epochs=1):
    writer = SummaryWriter('runs/resnet34_experiment')
    
    # Optional: Log model graph to TensorBoard
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    writer.add_graph(model, dummy_input)

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train() 
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                avg_step_loss = running_loss / 100
                writer.add_scalar('Loss/train_step', avg_step_loss, epoch * len(train_loader) + i)
                running_loss = 0.0 

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Optimized Memory management for your M3
        if device == 'mps':
            torch.mps.empty_cache()
        gc.collect()
            
        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_loader)

        # Update Scheduler (once per epoch)
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}%')

    writer.close()
    print("Training Complete!")    

    # Only saving the weights of the model, not the entire architecture, since we can easily reconstruct it with the code.  
    # Use .module if DataParallel is active, otherwise just model
    state_dict_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict_to_save, 'models/resnet34_cifar10_v1.pth')
    print("Model saved.")    



if __name__ == "__main__":

    # Parse User Arguments
    parser = argparse.ArgumentParser(description="Train ResNet-34 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=128, help="Image resolution (e.g., 32, 64, 128)")
    args = parser.parse_args()

    # Setup Data (using your dataset.py factory)
    train_loader, test_loader, classes = get_dataloaders(dataset_name="cifar10", batch_size=args.batch_size, image_size=args.image_size)
    
    # Setup Model and Device
    model, criterion, optimizer, scheduler, device = model_setup(num_classes=len(classes))
    
    # Launch Training
    train(model, criterion, optimizer, scheduler, args.image_size, train_loader, test_loader, device, num_epochs=args.epochs)

    # You can run this script from the command line like this:
    #`uv run python -m src.train --epochs 1 --image_size 128`