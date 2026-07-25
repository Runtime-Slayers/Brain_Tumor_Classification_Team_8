import torch
import torch.nn as nn
from dataset import get_dataloaders
from model import AdvancedTumorCNN
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import label_binarize

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, test_loader, classes = get_dataloaders("Dataset", batch_size=32, use_clahe=True)
    
    model = AdvancedTumorCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Classification Report
    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    all_labels_bin = label_binarize(all_labels, classes=[0,1,2,3])
    all_probs = np.array(all_probs)
    
    plt.figure(figsize=(8,6))
    for i in range(4):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    print("Evaluation complete. Saved confusion_matrix.png and roc_curve.png")

if __name__ == "__main__":
    evaluate_model()
