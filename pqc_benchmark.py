import oqs
import time
import matplotlib.pyplot as plt
import numpy as np

# Benchmark Kyber algorithm
algorithms = ['Kyber512', 'Kyber768', 'Kyber1024']
results = {}

for alg in algorithms:
    kem = oqs.KeyEncapsulation(alg)
    keygen_times = []
    enc_times = []
    dec_times = []
    # Run benchmarks
    for _ in range(100):  # 100 iterations for averaging
        # Key Generation
        start = time.time()
        public_key = kem.generate_keypair()
        keygen_times.append(time.time() - start)

        # Encapsulation (Encryption)
        start = time.time()
        ciphertext, shared_secret_enc = kem.encap_secret(public_key)
        enc_times.append(time.time() - start)

        # Decapsulation (Decryption)
        start = time.time()
        shared_secret_dec = kem.decap_secret(ciphertext)
        dec_times.append(time.time() - start)

    # Store results
    results[alg] = {
        'keygen_avg': np.mean(keygen_times) * 1000,  # Convert to ms
        'enc_avg': np.mean(enc_times) * 1000,
        'dec_avg': np.mean(dec_times) * 1000
    }

    kem.free()

# Display results
for alg, data in results.items():
    print(f"{alg}: KeyGen={data['keygen_avg']:.3f} ms, Enc={data['enc_avg']:.3f} ms, Dec={data['dec_avg']:.3f} ms")

# Plot results
alg_names = list(results.keys())
keygen_times = [results[alg]['keygen_avg'] for alg in alg_names]
enc_times = [results[alg]['enc_avg'] for alg in alg_names]
dec_times = [results[alg]['dec_avg'] for alg in alg_names]

x = np.arange(len(alg_names))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, keygen_times, width, label='KeyGen')
rects2 = ax.bar(x, enc_times, width, label='Encryption')
rects3 = ax.bar(x + width, dec_times, width, label='Decryption')

ax.set_xlabel('Algorithm')
ax.set_ylabel('Time (ms)')
ax.set_title('Post-Quantum Cryptographic Performance Benchmark')
ax.set_xticks(x)
ax.set_xticklabels(alg_names, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()
