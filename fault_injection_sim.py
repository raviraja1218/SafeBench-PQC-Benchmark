import oqs
import random
import numpy as np
import matplotlib.pyplot as plt

# Initialize Kyber for fault injection simulation
kem = oqs.KeyEncapsulation('Kyber512')
public_key = kem.generate_keypair()

# Fault Injection Function
def inject_fault(data):
    # Randomly flip a bit in the byte array
    byte_index = random.randint(0, len(data) - 1)
    bit_index = random.randint(0, 7)
    corrupted_byte = data[byte_index] ^ (1 << bit_index)
    return data[:byte_index] + bytes([corrupted_byte]) + data[byte_index + 1:]

# Simulation Parameters
num_trials = 1000
successful_decryptions = 0
faulty_decryptions = 0

# Simulation Loop
for _ in range(num_trials):
    ciphertext, shared_secret_original = kem.encap_secret(public_key)

    # Inject fault in ciphertext
    faulty_ciphertext = inject_fault(ciphertext)

    # Attempt to decapsulate the faulty ciphertext
    try:
        shared_secret_faulty = kem.decap_secret(faulty_ciphertext)
        if shared_secret_faulty == shared_secret_original:
            successful_decryptions += 1
        else:
            faulty_decryptions += 1
    except Exception:
        faulty_decryptions += 1  # Decapsulation failed due to fault

kem.free()

# Results
fault_rate = (faulty_decryptions / num_trials) * 100
success_rate = (successful_decryptions / num_trials) * 100

print(f"Successful Decryptions: {successful_decryptions}")
print(f"Faulty Decryptions: {faulty_decryptions}")
print(f"Fault Injection Success Rate: {fault_rate:.2f}%")
print(f"Decryption Resilience: {success_rate:.2f}%")

# Visualization
labels = ['Successful Decryptions', 'Faulty Decryptions']
values = [successful_decryptions, faulty_decryptions]

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
plt.title('Fault Injection Attack Simulation on Kyber512')
plt.show()
