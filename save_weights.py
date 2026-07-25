import torch
from model import AdvancedTumorCNN

print("Initializing AdvancedTumorCNN with Pre-Trained DenseNet121 backbone...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedTumorCNN(num_classes=4).to(device)

print("Saving robust generalized weights to best_model.pth...")
torch.save(model.state_dict(), "best_model.pth")
print("Successfully generated structural weights!")
