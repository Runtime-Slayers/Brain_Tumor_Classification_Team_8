import os
import torch
import matplotlib.pyplot as plt
from dataset import BrainTumorDataset
import numpy as np
from torchvision import transforms
from PIL import Image

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    dataset = BrainTumorDataset(root_dir='Dataset', split='Training', transform=transform, use_clahe=False)
    
    # Pick 10 random images
    indices = np.random.choice(len(dataset), 10, replace=False)
    
    fig, axes = plt.subplots(10, 2, figsize=(8, 30))
    
    for i, idx in enumerate(indices):
        img_path = dataset.image_paths[idx]
        orig_img = np.array(Image.open(img_path).convert('RGB'))
        
        # Preprocessed
        img, mask, label = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(f"Original: {dataset.classes[label]}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_np)
        axes[i, 1].set_title(f"Preprocessed (Hole-Filled)")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig('skull_strip_holes_verification.png')
    print("Saved skull_strip_holes_verification.png")

if __name__ == "__main__":
    main()
