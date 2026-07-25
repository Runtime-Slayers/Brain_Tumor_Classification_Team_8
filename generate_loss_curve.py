import matplotlib.pyplot as plt

train_loss = [0.7848, 0.5212, 0.4017, 0.3261, 0.2761, 0.3592, 0.3267, 0.2891, 0.2617, 0.2162, 0.1922, 0.1553, 0.1257, 0.1074, 0.0976, 0.2393, 0.1978, 0.1747, 0.1614, 0.1460]
val_loss = [1.0202, 1.1447, 1.3067, 0.5715, 0.5210, 0.5692, 0.7307, 0.4779, 0.4560, 0.6338, 0.4643, 0.6718, 0.4894, 0.4142, 0.4125, 2.3166, 0.5384, 0.7548, 0.7145, 0.4740]
epochs = range(1, 21)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
plt.title('Model Loss Curve (Early Stopping at Epoch 20)', fontsize=14, weight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Categorical Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('loss_curve.png')
print("loss_curve.png generated.")
