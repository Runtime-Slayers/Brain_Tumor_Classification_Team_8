import matplotlib.pyplot as plt
import numpy as np

# Absolute Raw Log Data
train_loss = [0.7848, 0.5212, 0.4017, 0.3261, 0.2761, 0.3592, 0.3267, 0.2891, 0.2617, 0.2162, 0.1922, 0.1553, 0.1257, 0.1074, 0.0976, 0.2393, 0.1978, 0.1747, 0.1614, 0.1460]
val_loss = [1.0202, 1.1447, 1.3067, 0.5715, 0.5210, 0.5692, 0.7307, 0.4779, 0.4560, 0.6338, 0.4643, 0.6718, 0.4894, 0.4142, 0.4125, 2.3166, 0.5384, 0.7548, 0.7145, 0.4740]
epochs = np.arange(1, 21)

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.append(a[:n-1], ret[n - 1:] / n)

train_ma = moving_average(train_loss, 3)
val_ma = moving_average(val_loss, 3)

plt.figure(figsize=(8, 6))
# 100% Authentic Reality (Faint)
plt.plot(epochs, train_loss, 'b-', alpha=0.25, label='Raw Train Loss (Reality)')
plt.plot(epochs, val_loss, 'r-', alpha=0.25, label='Raw Val Loss (Reality)')
# Statistical Trend (Bold)
plt.plot(epochs, train_ma, 'b-', linewidth=2.5, label='Train Trend (Moving Average)')
plt.plot(epochs, val_ma, 'r-', linewidth=2.5, label='Val Trend (Moving Average)')

plt.title('Loss Curve: Raw SGDR Reality vs Statistical Trend', fontsize=14, weight='bold')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Categorical Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('loss_curve.png')
print("Loss curve generated with Moving Average.")
