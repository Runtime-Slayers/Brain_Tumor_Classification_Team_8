import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

def perform_eda(dataset_path):
    print(f"Starting EDA on dataset at: {dataset_path}")
    splits = ['Training', 'Testing']
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    data = []
    
    # Track corrupted images
    corrupted_images = []

    for split in splits:
        for cls in classes:
            folder_path = os.path.join(dataset_path, split, cls)
            if not os.path.exists(folder_path):
                print(f"Directory not found: {folder_path}")
                continue
                
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                try:
                    # Check if file is an image and can be opened
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                        format = img.format
                        
                        data.append({
                            'Split': split,
                            'Class': cls,
                            'Width': width,
                            'Height': height,
                            'Mode': mode,
                            'Format': format,
                            'Path': img_path
                        })
                except Exception as e:
                    corrupted_images.append(img_path)
    
    df = pd.DataFrame(data)
    
    print("\n--- Basic Statistics ---")
    print(f"Total Images: {len(df)}")
    print(f"Corrupted Images Found: {len(corrupted_images)}")
    
    if len(corrupted_images) > 0:
        print("Corrupted Images:")
        for path in corrupted_images:
            print(path)
            
    # Save the dataframe to a CSV for reporting
    df.to_csv("eda_results.csv", index=False)
    
    # --- Plotting Class Distribution ---
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Class', hue='Split')
    plt.title('Class Distribution across Training and Testing Splits')
    plt.ylabel('Number of Images')
    plt.xlabel('Brain Tumor Class')
    plt.savefig('class_distribution.png')
    plt.close()
    
    # --- Plotting Image Size Distribution ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Width', y='Height', hue='Split', alpha=0.5)
    plt.title('Image Size Distribution')
    plt.xlabel('Image Width (Pixels)')
    plt.ylabel('Image Height (Pixels)')
    plt.savefig('image_size_distribution.png')
    plt.close()
    
    print("\n--- Summary by Class and Split ---")
    summary = df.groupby(['Split', 'Class']).size().unstack(fill_value=0)
    print(summary)
    
    # --- Pixel Intensity Statistics (Sampling 10 images per class from training) ---
    plt.figure(figsize=(12, 8))
    for i, cls in enumerate(classes, 1):
        sample_paths = df[(df['Class'] == cls) & (df['Split'] == 'Training')].sample(n=min(10, len(df)), random_state=42)['Path'].values
        pixel_values = []
        for path in sample_paths:
            img = Image.open(path).convert('L') # convert to grayscale
            pixel_values.extend(np.array(img).flatten())
        
        plt.subplot(2, 2, i)
        plt.hist(pixel_values, bins=50, color='gray', alpha=0.7)
        plt.title(f'Pixel Intensity - {cls}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
    plt.tight_layout()
    plt.savefig('pixel_intensity_histograms.png')
    plt.close()
    
    print("\nEDA complete. Generated plots: class_distribution.png, image_size_distribution.png, pixel_intensity_histograms.png")
    
if __name__ == "__main__":
    perform_eda("Dataset")
