import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, split='Training', transform=None, use_clahe=True):
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.transform = transform
        self.use_clahe = use_clahe
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]
        
        # Clean slate: No skull stripping, elastic deformation, or blurs.
        # We rely on AG-ResNet34's built-in attention to organically find the ROI.
        img = Image.fromarray(image)
        
        if self.use_clahe:
            lab = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = Image.fromarray(cv2.cvtColor(limg, cv2.COLOR_LAB2RGB))

        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_dataloaders(root_dir, batch_size=32, img_size=224, use_clahe=True):
    # Minimal safe augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(5),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = BrainTumorDataset(root_dir, split='Training', transform=train_transform, use_clahe=use_clahe)
    test_dataset = BrainTumorDataset(root_dir, split='Testing', transform=test_transform, use_clahe=use_clahe)
    
    # num_workers=0 to prevent multiprocessing freezes on Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, test_loader, train_dataset.classes
