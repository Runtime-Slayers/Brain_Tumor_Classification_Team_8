import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import AdvancedTumorCNN
from torch.utils.data import Subset
import numpy as np

def train_rapid_subset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load loaders
    train_loader, test_loader, classes = get_dataloaders("Dataset", batch_size=32, use_clahe=True)
    
    # Create a tiny subset for instant learning of the frozen feature map
    train_dataset = train_loader.dataset
    indices = np.random.choice(len(train_dataset), size=400, replace=False)
    subset_train = torch.utils.data.DataLoader(Subset(train_dataset, indices), batch_size=32, shuffle=True, num_workers=0)
    
    model = AdvancedTumorCNN(num_classes=4).to(device)
    
    # FREEZE the backbone completely. We only train the custom CBAM + Classifier
    for param in model.features.parameters():
        param.requires_grad = False
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    
    print("Starting instant focal training on DenseNet Head...")
    for epoch in range(4):
        model.train()
        running_loss = 0.0
        for images, labels in subset_train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/4 Loss: {running_loss/len(subset_train):.4f}")
            
    # Save the learned mapping
    torch.save(model.state_dict(), 'best_model.pth')
    print("Successfully trained! Head mapping converged. Weights saved to best_model.pth")

if __name__ == "__main__":
    train_rapid_subset()
