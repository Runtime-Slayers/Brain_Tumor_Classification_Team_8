import torch
import matplotlib.pyplot as plt
from dataset import BrainTumorDataset
import numpy as np
from torchvision import transforms

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    dataset = BrainTumorDataset(root_dir='Dataset', split='Training', transform=transform, use_clahe=False)
    
    # Pick 10 random images
    indices = np.random.choice(len(dataset), 10, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img, mask, label = dataset[idx]
        
        # img is a tensor [3, H, W]
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Class: {dataset.classes[label]}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('skull_strip_verification.png')
    print("Saved skull_strip_verification.png")

if __name__ == "__main__":
    main()
