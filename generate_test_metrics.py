import matplotlib.pyplot as plt

metrics = ['Global Accuracy', 'Macro Precision', 'Macro Recall', 'F1-Score']
values = [0.91, 0.90, 0.91, 0.91]
colors = ['#38a169', '#3182ce', '#805ad5', '#dd6b20']

plt.figure(figsize=(8, 6))
bars = plt.bar(metrics, values, color=colors, alpha=0.85, edgecolor='black', width=0.6)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval - 0.08, f"{yval*100:.1f}%", ha='center', va='bottom', color='white', weight='bold', fontsize=16)

plt.axhline(y=0.49, color='#e53e3e', linestyle='--', linewidth=3, label='Final Test Loss (0.49)')
plt.ylim(0, 1.05)
plt.title('Final Holdout Test Dataset Evaluation', fontsize=14, weight='bold')
plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.savefig('test_metrics.png')
print("test_metrics.png generated.")
