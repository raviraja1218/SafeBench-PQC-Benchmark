import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import oqs  # Post-Quantum Cryptography Library
import time
import foolbox as fb  # For adversarial attacks
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# ---- 1️⃣ Load CIFAR-10 Dataset ----
transform = transforms.Compose([
    transforms.ToTensor(),
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# ---- 2️⃣ Load Pretrained AI Model (ResNet-18) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10
model = model.to(device)
model.eval()


# ---- 3️⃣ Apply Kyber512 PQC Encryption to Image Data ----
def encrypt_image(image, kem):
    public_key = kem.generate_keypair()
    ciphertext, _ = kem.encap_secret(public_key)

    # Ensure ciphertext matches the image size
    ciphertext_expanded = np.tile(np.frombuffer(ciphertext, dtype=np.uint8), (3072 // len(ciphertext) + 1))[:3072]

    encrypted_image = image + torch.tensor(ciphertext_expanded.reshape(image.shape)).float() / 255.0
    return encrypted_image


kem = oqs.KeyEncapsulation('Kyber512')


# ---- 4️⃣ Test Model Accuracy on Normal & PQC-Encrypted Data ----
def evaluate(model, dataloader, pqc_encrypt=False):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            if pqc_encrypt:
                images = torch.stack([encrypt_image(img, kem) for img in images])
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


baseline_acc = evaluate(model, testloader, pqc_encrypt=False)
pqc_acc = evaluate(model, testloader, pqc_encrypt=True)
print(f"Baseline Accuracy: {baseline_acc:.2f}%")
print(f"PQC Encrypted Accuracy: {pqc_acc:.2f}%")


# ---- 5️⃣ Define Adversarial Attack Function ----
def attack_model(model, dataloader, attack_method, epsilons=0.03):
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = attack_method()

    correct = 0
    total = 0

    for i, (images, labels) in enumerate(dataloader):
        if i >= 5:  # Process only the first 5 batches for speed
            break

        images, labels = images.to(device), labels.to(device)

        # Apply the attack and extract only the adversarial images
        adversarial_images, _, _ = attack(fmodel, images, labels, epsilons=epsilons)

        outputs = model(adversarial_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ---- 6️⃣ Apply FGSM, PGD, and Quantum Perturbation Attacks ----
fgsm_acc = attack_model(model, testloader, fb.attacks.FGSM, epsilons=0.03)
pgd_acc = attack_model(model, testloader, fb.attacks.PGD, epsilons=0.03)
quantum_attack_acc = pqc_acc  # Using PQC encryption as an adversarial perturbation

# ---- 7️⃣ Display Final Results ----
print("\nFinal AI Robustness Results:")
print(f"Baseline Accuracy: {baseline_acc:.2f}%")
print(f"PQC Encrypted Accuracy: {pqc_acc:.2f}%")
print(f"FGSM Attack Accuracy: {fgsm_acc:.2f}%")
print(f"PGD Attack Accuracy: {pgd_acc:.2f}%")
print(f"Quantum-PQC Perturbation Accuracy: {quantum_attack_acc:.2f}%")

# ---- 8️⃣ Free PQC Resources ----
kem.free()
