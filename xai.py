import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import AdvancedTumorCNN
from PIL import Image
from torchvision import transforms

def get_gradcam(image_path, model_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = AdvancedTumorCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Choose target layer for Grad-CAM.
    # In AdvancedTumorCNN, the most semantic feature map is right before the global pooling,
    # which is the spatial attention conv layer inside final_cbam.
    target_layers = [model.final_cbam.sa.conv1]
    
    # Preprocess Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Optional CLAHE for visualization consistency
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    img_resized = cv2.resize(img_clahe, (224, 224))
    rgb_img = np.float32(img_resized) / 255
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(Image.fromarray(img_resized)).unsqueeze(0).to(device)
    
    # Forward pass to get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = outputs.argmax(dim=1).item()
        
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_label = classes[predicted_class]
    
    # Construct the CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Define targets
    targets = [ClassifierOutputTarget(predicted_class)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay CAM on original image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Save Image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original (CLAHE) Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"Grad-CAM (Pred: {predicted_label})")
    plt.axis('off')
    
    plt.savefig(output_path)
    plt.close()
    
    return predicted_label, cam_image

if __name__ == "__main__":
    # Test on a sample image (replace with actual path during testing)
    import os
    import glob
    
    test_images = glob.glob("Dataset/Testing/*/*.jpg")
    if len(test_images) > 0:
        sample_img = test_images[0]
        get_gradcam(sample_img, "best_model.pth", "sample_gradcam.png")
        print("Generated sample_gradcam.png")
    else:
        print("No test images found for testing XAI.")
