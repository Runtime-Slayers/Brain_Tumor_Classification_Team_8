import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import AdvancedTumorCNN
import matplotlib.pyplot as plt

def train_fast():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader, classes = get_dataloaders("Dataset", batch_size=32, use_clahe=True)
    
    model = AdvancedTumorCNN(num_classes=4).to(device)
    
    # FREEZE the DenseNet backbone for insanely fast training (only train CBAM + Classifier)
    for param in model.features.parameters():
        param.requires_grad = False
        
    criterion = nn.CrossEntropyLoss()
    # High learning rate since we are only training the head
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    
    print("Starting rapid transfer learning...")
    best_val_acc = 0.0
    
    for epoch in range(3):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/3 - Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc or epoch == 0:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    print("Transfer learning complete. New best_model.pth generated successfully.")

if __name__ == "__main__":
    train_fast()
