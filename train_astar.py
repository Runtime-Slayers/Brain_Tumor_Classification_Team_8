import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import TumorFocusCNN
import heapq
import numpy as np
import copy
import matplotlib.pyplot as plt

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def a_star_hyperparameter_search(train_loader, test_loader, device):
    """
    A-Star Search for Hyperparameter Optimization.
    We define a grid of learning rates and dropout values.
    Cost g(n) = Validation Loss
    Heuristic h(n) = 0 (Uniform Cost Search/Dijkstra variant for optimization)
    We expand the best nodes to find the local minimum.
    """
    print("Starting A-Star Hyperparameter Optimization...")
    lr_grid = [1e-4, 5e-4, 1e-3, 2e-3]
    dropout_grid = [0.3, 0.4, 0.5, 0.6]
    
    # Start at the center of the grid
    start_node = (1, 2) # (5e-4, 0.5)
    
    # Priority Queue for A*: stores (f_cost, node, model_state, val_acc)
    pq = []
    visited = set()
    
    def evaluate_node(node):
        lr = lr_grid[node[0]]
        dropout = dropout_grid[node[1]]
        print(f"Evaluating Node {node}: LR={lr}, Dropout={dropout}")
        
        model = TumorFocusCNN(num_classes=4).to(device)
        model.dropout = nn.Dropout(dropout) # Modify dropout
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train for 2 epochs as a proxy for fitness (Heuristic evaluation)
        for epoch in range(2):
            train_epoch(model, train_loader, criterion, optimizer, device)
            
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Node {node} Result - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        return val_loss, val_acc, model.state_dict()
    
    # Evaluate start node
    val_loss, val_acc, state = evaluate_node(start_node)
    visited.add(start_node)
    # g(n) = val_loss, h(n) = 0 -> f(n) = val_loss
    heapq.heappush(pq, (val_loss, start_node, state, val_acc))
    
    best_node = start_node
    best_loss = val_loss
    best_state = state
    
    # Max expansions
    max_expansions = 5
    expansions = 0
    
    while pq and expansions < max_expansions:
        current_loss, current_node, current_state, current_acc = heapq.heappop(pq)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_node = current_node
            best_state = current_state
            
        # Get neighbors (up, down, left, right in grid)
        neighbors = [
            (current_node[0]-1, current_node[1]),
            (current_node[0]+1, current_node[1]),
            (current_node[0], current_node[1]-1),
            (current_node[0], current_node[1]+1)
        ]
        
        for neighbor in neighbors:
            if 0 <= neighbor[0] < len(lr_grid) and 0 <= neighbor[1] < len(dropout_grid):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_loss, n_acc, n_state = evaluate_node(neighbor)
                    heapq.heappush(pq, (n_loss, neighbor, n_state, n_acc))
                    
        expansions += 1
        
    best_lr = lr_grid[best_node[0]]
    best_dropout = dropout_grid[best_node[1]]
    print(f"\nA-Star Search Complete! Best Hyperparameters: LR={best_lr}, Dropout={best_dropout}")
    
    return best_lr, best_dropout

def full_training(best_lr, best_dropout, train_loader, test_loader, device, num_epochs=10):
    print(f"\nStarting Full Training with Best Hyperparameters...")
    model = TumorFocusCNN(num_classes=4).to(device)
    model.dropout = nn.Dropout(best_dropout)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.savefig('training_curves.png')
    plt.close()
    
    print("Training complete. Best model saved to 'best_model.pth'. Curves saved to 'training_curves.png'.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use small batch size for A-Star to run faster
    train_loader, test_loader, classes = get_dataloaders("Dataset", batch_size=32, use_clahe=True)
    
    # 1. A-Star Search (Proxy optimization)
    best_lr, best_dropout = a_star_hyperparameter_search(train_loader, test_loader, device)
    
    # 2. Full Training
    full_training(best_lr, best_dropout, train_loader, test_loader, device, num_epochs=10)
