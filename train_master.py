import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import AdvancedTumorCNN
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import time
import matplotlib.patches as patches

def generate_live_telemetry(all_labels, all_preds, all_probs, all_losses, classes, epoch):
    num_classes = len(classes)
    
    # 1. Live Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=[c.upper() for c in classes], yticklabels=[c.upper() for c in classes], ax=ax)
    
    idx_m = classes.index('meningioma') if 'meningioma' in classes else 1
    idx_p = classes.index('pituitary') if 'pituitary' in classes else 3
    ax.add_patch(patches.Rectangle((idx_p, idx_m), 1, 1, fill=False, edgecolor='red', lw=3))
    ax.add_patch(patches.Rectangle((idx_m, idx_p), 1, 1, fill=False, edgecolor='red', lw=3))
    
    plt.title(f'Live Normalized CM (Epoch {epoch+1})', weight='bold')
    plt.ylabel('True Pathology')
    plt.xlabel('AI Predicted Pathology')
    plt.tight_layout()
    plt.savefig('live_cm.png')
    plt.close()
    
    # 2. Loss Distribution
    loss_arr = np.array(all_losses)
    lbl_arr = np.array(all_labels)
    mean_losses = [loss_arr[lbl_arr == i].mean() if np.sum(lbl_arr == i) > 0 else 0 for i in range(num_classes)]
    plt.figure(figsize=(6, 5))
    plt.bar([c.upper() for c in classes], mean_losses, color=['#2b6cb0', '#e53e3e', '#38a169', '#ecc94b'])
    plt.title('Validation CE Loss', weight='bold')
    plt.ylabel('Mean Loss')
    plt.tight_layout()
    plt.savefig('focal_loss_dist.png') # keeping filename same for UI compatibility
    plt.close()
    
    # 3. Margin Score Tracker
    all_probs_tensor = torch.tensor(np.array(all_probs))
    top2, _ = torch.topk(all_probs_tensor, 2, dim=1)
    margins = (top2[:, 0] - top2[:, 1]).numpy()
    plt.figure(figsize=(6, 5))
    plt.hist(margins, bins=20, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Softmax Margin Score (Predictive Entropy)', weight='bold')
    plt.xlabel('Margin (Top 1 Prob - Top 2 Prob)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('margin_score.png')
    plt.close()

def train_master():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Starting ResNet34 Fine-Tuning)")
    
    train_loader, test_loader, classes = get_dataloaders("Dataset", batch_size=32, use_clahe=True)
    num_classes = len(classes)
    
    model = AdvancedTumorCNN(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    criterion_unreduced = nn.CrossEntropyLoss(reduction='none')
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    
    train_losses, val_losses, val_accs = [], [], []
    best_val_loss = float('inf')
    
    print("\n--- AG-ResNet34 Deep Fine-Tuning ---")
    
    for epoch in range(8):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_labels, all_probs, all_preds, all_batch_losses = [], [], [], []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss_batch = criterion_unreduced(outputs, labels)
                val_loss += loss_batch.sum().item()
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_batch_losses.extend(loss_batch.cpu().numpy())
                
        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = correct / total
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        scheduler.step()
        
        generate_live_telemetry(all_labels, all_preds, all_probs, all_batch_losses, classes, epoch)
        
        print(f"Epoch {epoch+1}/8 | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Time: {time.time()-start_time:.1f}s")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  --> New best model saved!")

    # ---------------------------------------------------------
    # METRICS GENERATION
    # ---------------------------------------------------------
    print("\n--- Generating Validation Graphs ---")
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss', color='#2b6cb0', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#e53e3e', linewidth=2)
    plt.title('Training Loss Curve', weight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('loss_curve.png', bbox_inches='tight')
    plt.close()
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[c.upper() for c in classes], yticklabels=[c.upper() for c in classes])
    plt.title('Clinical Confusion Matrix', weight='bold')
    plt.ylabel('True Pathology')
    plt.xlabel('AI Predicted Pathology')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()
    
    all_labels_bin = label_binarize(all_labels, classes=[0,1,2,3])
    all_probs = np.array(all_probs)
    plt.figure(figsize=(8, 6))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
    for i, color in zip(range(num_classes), colors):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{classes[i].upper()} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)', weight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png', bbox_inches='tight')
    plt.close()
    
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    class_names = [c.upper() for c in classes]
    scores = np.array([[report[c][m] for m in metrics] for c in classes])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, scores[:, 0]*100, width, label='Precision', color='#2b6cb0')
    ax.bar(x, scores[:, 1]*100, width, label='Recall', color='#e53e3e')
    ax.bar(x + width, scores[:, 2]*100, width, label='F1-Score', color='#38a169')
    ax.set_ylabel('Score (%)', weight='bold')
    ax.set_title('Pathology Evaluation Metrics', weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('test_metrics.png', bbox_inches='tight')
    plt.close()
    
    print("Master training complete. All curves saved correctly.")

if __name__ == "__main__":
    train_master()
