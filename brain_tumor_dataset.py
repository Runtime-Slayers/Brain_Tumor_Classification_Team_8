"""
=============================================================================
BRAIN TUMOR CLASSIFICATION — TEAM 8
File: brain_tumor_dataset.py
Role: Dataset Loading, Preprocessing & Augmentation Pipeline
=============================================================================

Author      : Team 8
Project     : Brain Tumor Classification using Deep Learning & Explainable AI
Architecture: Attention-Gated ResNet-34 with Monte Carlo Uncertainty Estimation
Dataset     : Multi-class MRI Brain Tumor Dataset
              (Glioma | Meningioma | Pituitary Tumor | No Tumor)

Description :
    This module defines the PyTorch Dataset class (BrainTumorDataset) and
    the DataLoader factory function (get_dataloaders). It handles:
      - CLAHE adaptive contrast enhancement for MRI preprocessing
      - Biophysically justified on-the-fly augmentation (training split only)
      - ImageNet-aligned normalization for ResNet transfer learning
      - Inverse-frequency class weight computation for imbalance compensation
=============================================================================
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet pre-training statistics used to normalize all input tensors.
# This ensures transfer learning features align with the ResNet-34 manifold.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Standard input resolution expected by ResNet-34 and most CNN backbones.
IMG_SIZE = 224

# CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters.
# Tile grid: 8×8 sub-blocks applied to LAB L-channel for local contrast boost.
# Clip Limit: 2.0 — prevents over-amplification of uniform background noise.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(pil_image: Image.Image) -> Image.Image:
    """
    Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE) to an
    input PIL RGB image to boost local tissue contrast in MRI scans.

    Clinical Justification:
        MRI scans suffer from field-strength-dependent intensity variation.
        Global normalization washes out critical tumor boundary detail.
        CLAHE amplifies local lesion texture contrast without over-enhancing
        background air or uniform white matter, improving feature separability
        in the convolutional feature space.

    Args:
        pil_image (PIL.Image): Input RGB PIL image from dataset loader.

    Returns:
        PIL.Image: CLAHE-enhanced image in RGB colour space.
    """
    # Convert RGB → LAB (device-independent perceptual colour space).
    # CLAHE is applied only to the L (Lightness) channel to avoid colour shift.
    img_np  = np.array(pil_image)
    lab     = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Create CLAHE operator and apply to luminance channel
    clahe_op = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    cl        = clahe_op.apply(l)

    # Reconstruct LAB image with enhanced L-channel and convert back to RGB
    lab_enhanced = cv2.merge((cl, a, b))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return Image.fromarray(rgb_enhanced)


def compute_class_weights(dataset: "BrainTumorDataset") -> torch.Tensor:
    """
    Compute inverse-frequency class weights to address class imbalance.

    Mathematical Formulation:
        w_i = N_total / (K × N_i)

        where:
            N_total  = total number of samples across all classes
            K        = number of unique classes
            N_i      = number of samples in class i

    Clinical Justification:
        The dataset contains a significant minority of 'No Tumor' scans (~15%).
        Without compensation, the Cross-Entropy loss minimizer will learn to
        over-predict neoplastic pathology. Inverse-frequency weighting forces
        the optimizer to penalize misclassifications on minority classes more
        severely, maintaining clinical sensitivity across all categories.

    Args:
        dataset (BrainTumorDataset): Instantiated training dataset object.

    Returns:
        torch.Tensor: 1D float tensor of per-class penalty weights,
                      usable directly in nn.CrossEntropyLoss(weight=...).
    """
    label_counts = Counter(dataset.labels)
    n_total      = len(dataset.labels)
    n_classes    = len(dataset.classes)

    weights = []
    for class_idx in range(n_classes):
        count = label_counts.get(class_idx, 1)                  # Avoid division by zero
        weight = n_total / (n_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for the Brain Tumor MRI classification task.

    Directory Structure Expected:
        Dataset/
          Training/
            glioma/        *.jpg | *.jpeg | *.png
            meningioma/    *.jpg | *.jpeg | *.png
            notumor/       *.jpg | *.jpeg | *.png
            pituitary/     *.jpg | *.jpeg | *.png
          Testing/
            (same structure as Training)

    Preprocessing applied per sample:
        1. Load image as PIL RGB (handles grayscale conversions internally).
        2. Apply CLAHE enhancement on LAB L-channel (if use_clahe=True).
        3. Apply split-specific torchvision transforms (resize, augment, normalize).

    Args:
        root_dir  (str)  : Root dataset directory (e.g. "Dataset").
        split     (str)  : "Training" or "Testing" — selects appropriate sub-directory.
        transform        : torchvision.transforms.Compose pipeline to apply.
        use_clahe (bool) : Whether to apply CLAHE preprocessing (default: True).
    """

    def __init__(self, root_dir: str, split: str = 'Training',
                 transform=None, use_clahe: bool = True):
        self.root_dir  = os.path.join(root_dir, split)
        self.split     = split
        self.transform = transform
        self.use_clahe = use_clahe

        # Discover class names from subdirectory structure
        self.classes      = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Enumerate all valid image paths and their corresponding integer labels
        self.image_paths: list[str] = []
        self.labels:      list[int] = []

        VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                ext = os.path.splitext(img_name)[-1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue                                      # Skip non-image files
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self) -> int:
        """Return total number of image samples in this dataset split."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Retrieve and preprocess a single sample by index.

        Args:
            idx (int): Integer index of the sample to retrieve.

        Returns:
            tuple: (image_tensor [C, H, W], label [int])
        """
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        # Load image as RGB (ensuring consistent 3-channel input even for grayscale scans)
        image = Image.open(img_path).convert('RGB')

        # Stage 1: CLAHE adaptive contrast normalization
        if self.use_clahe:
            image = apply_clahe(image)

        # Stage 2: Apply torchvision transforms (resize, augmentation, tensor conversion, normalize)
        if self.transform:
            image = self.transform(image)

        return image, label


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(root_dir: str,
                    batch_size: int = 32,
                    img_size:   int = IMG_SIZE,
                    use_clahe:  bool = True):
    """
    Construct and return PyTorch DataLoaders for training and test splits.

    Augmentation Strategy (Training Split Only):
        - RandomHorizontalFlip: Simulates bilateral brain symmetry variation
          (biophysically valid — left/right cortex are neurologically symmetric).
        - RandomRotation(±15°): Mimics imperfect patient head fixation inside MRI coil.
        - Bilinear Resize to 256×256 → CenterCrop to 224×224: Adds mild scale variation.

        EXCLUDED:
        - Vertical Flip: Inverted cerebral anatomy never occurs in standard protocol scans.
        - Colour Jitter: MRI T1 contrast physics are grayscale-only and channel-agnostic.
        - Extreme shearing (>20°): Structurally implausible for cranial acquisition.

    Normalization:
        All tensors are normalized to ImageNet statistics (mean/std per RGB channel)
        to maximally exploit ResNet-34 pre-trained transfer learning features.

    Args:
        root_dir   (str)  : Path to root dataset directory.
        batch_size (int)  : Mini-batch size for SGD training (default: 32).
        img_size   (int)  : Target spatial dimension after crop (default: 224).
        use_clahe  (bool) : Apply CLAHE preprocessing in Dataset (default: True).

    Returns:
        tuple: (train_loader, test_loader, class_names, class_weights)
            - train_loader   : DataLoader for Training split with augmentation.
            - test_loader    : DataLoader for Testing split with no augmentation.
            - class_names    : Sorted list of class name strings.
            - class_weights  : Inverse-frequency class weights tensor for weighted loss.
    """

    # ── Training transforms: augmentation + normalization ──────────────────
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),                              # Slightly upscale to allow random crop
        transforms.RandomHorizontalFlip(p=0.5),                    # Bilateral symmetry flip
        transforms.RandomRotation(degrees=15),                      # Simulate patient head tilt
        transforms.CenterCrop(img_size),                            # Crop to 224×224
        transforms.ToTensor(),                                      # Convert to [0,1] float tensor [C,H,W]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) # ImageNet standardization
    ])

    # ── Test transforms: deterministic pipeline, NO augmentation ───────────
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Instantiate Dataset objects
    train_dataset = BrainTumorDataset(root_dir, split='Training', transform=train_transform, use_clahe=use_clahe)
    test_dataset  = BrainTumorDataset(root_dir, split='Testing',  transform=test_transform,  use_clahe=use_clahe)

    # Compute inverse-frequency class weights for imbalanced loss correction
    class_weights = compute_class_weights(train_dataset)

    # ── DataLoader construction ─────────────────────────────────────────────
    # num_workers=0: Required on Windows to prevent multiprocessing deadlocks
    # with PyTorch's DataLoader spawn mode.
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"[Dataset] Training samples : {len(train_dataset)}")
    print(f"[Dataset] Testing samples  : {len(test_dataset)}")
    print(f"[Dataset] Classes          : {train_dataset.classes}")
    print(f"[Dataset] Class weights    : {class_weights.numpy().round(4)}")

    return train_loader, test_loader, train_dataset.classes, class_weights


# ─────────────────────────────────────────────────────────────────────────────
# QUICK VALIDATION ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Brain Tumor Dataset — Preprocessing Validation")
    print("=" * 60)

    train_loader, test_loader, classes, weights = get_dataloaders(
        root_dir="Dataset", batch_size=8, use_clahe=True
    )

    # Fetch a single batch to validate tensor shapes and types
    images, labels = next(iter(train_loader))
    print(f"\nSample batch — Image tensor : {images.shape}")   # Expected: [8, 3, 224, 224]
    print(f"Sample batch — Labels       : {labels.tolist()}")
    print(f"Pixel value range           : [{images.min():.3f}, {images.max():.3f}]")
    print(f"\nClass index mapping: {dict(zip(classes, range(len(classes))))}")
    print(f"Class weights (loss penalty): {weights.numpy().round(4)}")
    print("\nDataset module validation complete.")
