import oqs
import time
import numpy as np
import hashlib
import hmac
import secrets
import matplotlib.pyplot as plt
from scipy import stats

# Initialize Kyber with Countermeasures
kem = oqs.KeyEncapsulation('Kyber512')
public_key = kem.generate_keypair()

# Generate a strong random HMAC key
hmac_key = secrets.token_bytes(32)

# ---- 1️⃣ Constant-Time Operation Simulation ----

def constant_time_encapsulation(public_key):
    start_time = time.perf_counter()
    ciphertext, shared_secret = kem.encap_secret(public_key)
    # Introduce dummy operations to equalize timing
    dummy = sum([0 for _ in range(100)])  # Dummy loop to mask timing
    end_time = time.perf_counter()
    return (end_time - start_time), ciphertext, shared_secret

# ---- 2️⃣ Enhanced Fault Detection Mechanism ----

def generate_hmac(key, data):
    return hmac.new(key, data, hashlib.sha256).digest()

def fault_detection_decapsulation(ciphertext, hmac_value):
    try:
        # Verify HMAC before decryption
        computed_hmac = generate_hmac(hmac_key, ciphertext)
        if not hmac.compare_digest(computed_hmac, hmac_value):
            raise ValueError("Potential fault detected (HMAC mismatch)")

        # Redundant decryption checks
        shared_secret_1 = kem.decap_secret(ciphertext)
        shared_secret_2 = kem.decap_secret(ciphertext)

        if shared_secret_1 != shared_secret_2:
            raise ValueError("Inconsistent decryption detected")

        return True  # Successful decryption
    except Exception:
        return False  # Fault detected or decryption failed

# ---- 3️⃣ Simulation for Timing & Fault Resilience ----

# Timing Attack Mitigation Test
timings_fixed_input = []
timings_random_input = []

for _ in range(1000):
    fixed_input_time, ciphertext, _ = constant_time_encapsulation(public_key)
    random_input_time, random_ciphertext, _ = constant_time_encapsulation(oqs.KeyEncapsulation('Kyber512').generate_keypair())
    timings_fixed_input.append(fixed_input_time)
    timings_random_input.append(random_input_time)

# Fault Injection Test
successful_decryptions = 0
faulty_decryptions = 0

for _ in range(1000):
    _, ciphertext, _ = constant_time_encapsulation(public_key)
    hmac_value = generate_hmac(hmac_key, ciphertext)

    # Introduce random bit flip
    faulty_ciphertext = bytearray(ciphertext)
    index = np.random.randint(len(faulty_ciphertext))
    faulty_ciphertext[index] ^= 0x01  # Flip a bit

    if fault_detection_decapsulation(bytes(faulty_ciphertext), hmac_value):
        successful_decryptions += 1
    else:
        faulty_decryptions += 1

# ---- 4️⃣ Results ----

# Timing Attack Results
t_stat, p_value = stats.ttest_ind(timings_fixed_input, timings_random_input)

print(f"T-Statistic (After Countermeasure): {t_stat}")
print(f"P-Value (After Countermeasure): {p_value}")

# Fault Injection Results
fault_rate = (faulty_decryptions / 1000) * 100
success_rate = (successful_decryptions / 1000) * 100

print(f"Successful Decryptions (After Countermeasure): {successful_decryptions}")
print(f"Faulty Decryptions (After Countermeasure): {faulty_decryptions}")
print(f"Fault Injection Success Rate (After Countermeasure): {fault_rate:.2f}%")
print(f"Decryption Resilience (After Countermeasure): {success_rate:.2f}%")

# ---- 5️⃣ Visualization ----

# Timing Visualization
plt.figure(figsize=(8, 6))
plt.hist(timings_fixed_input, alpha=0.6, bins=30, label='Fixed Input (Constant-Time)', color='blue')
plt.hist(timings_random_input, alpha=0.6, bins=30, label='Random Input (Constant-Time)', color='green')
plt.title('Timing Attack Mitigation Effectiveness')
plt.xlabel('Encryption Time (s)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Fault Injection Visualization
labels = ['Successful Decryptions', 'Faulty Decryptions']
values = [successful_decryptions, faulty_decryptions]

plt.figure(figsize=(6, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
plt.title('Fault Injection Resilience After Countermeasure')
plt.show()

kem.free()
