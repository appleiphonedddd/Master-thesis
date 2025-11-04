import matplotlib.pyplot as plt

a_values = [0.1, 0.3, 0.9]
fedas_acc = [97.74, 94.81, 92.58]
fedinf_acc = [97.36, 92.58, 89.52]

plt.figure(figsize=(8, 5))

plt.plot(a_values, fedas_acc, marker='o', color='blue', label='FedAS')
plt.plot(a_values, fedinf_acc, marker='s', color='red', label='FedINF')

plt.title('Accuracy Comparison under Different Dirichlet α On FashionMNIST', fontsize=14)
plt.xlabel('α', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

plt.xticks(a_values)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

for x, y in zip(a_values, fedas_acc):
    plt.text(x, y + 0.05, f'{y:.2f}%', ha='center', color='blue')
for x, y in zip(a_values, fedinf_acc):
    plt.text(x, y - 0.3, f'{y:.2f}%', ha='center', color='red')

plt.tight_layout()
plt.savefig('Dirichlet_accuracy_comparison.png', dpi=300)
print("[Info] Saved Dirichlet accuracy comparison plot to Dirichlet_accuracy_comparison.png")
