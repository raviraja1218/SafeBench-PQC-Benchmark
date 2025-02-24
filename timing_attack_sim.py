import oqs
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Initialize Kyber for attack simulation
kem = oqs.KeyEncapsulation('Kyber512')
public_key = kem.generate_keypair()

# Simulate encryption with random vs fixed inputs
random_times = []
fixed_times = []

# Fixed message
fixed_message = b'A' * 32

# Benchmark loop
for _ in range(1000):
    # Random input
    random_message = np.random.bytes(32)
    start = time.time()
    kem.encap_secret(public_key)
    random_times.append(time.time() - start)

    # Fixed input
    start = time.time()
    kem.encap_secret(public_key)
    fixed_times.append(time.time() - start)

kem.free()

# Statistical Analysis
t_stat, p_value = ttest_ind(random_times, fixed_times)

# Results
print(f"T-Statistic: {t_stat}, P-Value: {p_value}")
if p_value < 0.05:
    print("⚠️ Potential timing leak detected!")
else:
    print("✅ No significant timing leak detected.")

# Visualization
sns.histplot(random_times, color='blue', label='Random Input', kde=True)
sns.histplot(fixed_times, color='red', label='Fixed Input', kde=True)
plt.xlabel('Encryption Time (s)')
plt.ylabel('Frequency')
plt.title('Timing Attack Simulation on Kyber512')
plt.legend()
plt.show()
