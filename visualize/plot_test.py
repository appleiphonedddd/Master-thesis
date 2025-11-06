import matplotlib.pyplot as plt

a_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
fedas_acc = [97.74, 95.07, 94.81, 93.81, 93.16, 92.64, 91.94, 91.88, 91.81]
fedinf_acc = [97.36, 93.48, 92.58, 91.24, 90.34, 89.25, 88.65, 88.60, 88.54]

plt.figure(figsize=(8, 5))

plt.plot(a_values, fedas_acc, marker='o', color='blue', label='FedAS')
plt.plot(a_values, fedinf_acc, marker='s', color='red', label='FedINF')

plt.title('Accuracy Comparison under Different Dirichlet α On FashionMNIST', fontsize=14)
plt.xlabel('α', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

plt.xticks(a_values)

plt.legend()

for x, y in zip(a_values, fedas_acc):
    plt.text(x, y + 0.05, f'{y:.2f}%', ha='center', color='blue')
for x, y in zip(a_values, fedinf_acc):
    plt.text(x, y - 0.3, f'{y:.2f}%', ha='center', color='red')

plt.tight_layout()
plt.savefig('Dirichlet_accuracy_FashionMNIST.png', dpi=300)
print("[Info] Saved Dirichlet accuracy comparison plot to Dirichlet_accuracy_FashionMNIST.png")
