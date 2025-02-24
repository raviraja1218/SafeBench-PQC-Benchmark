# 📜 SafeBench-PQC-Benchmark: Post-Quantum Cryptography-Enhanced AI Robustness Benchmark

**🔐 Project Focus:**  
SafeBench-PQC-Benchmark is a **Post-Quantum Cryptography (PQC)-enhanced AI robustness benchmarking framework**. It evaluates AI security against classical adversarial attacks (**FGSM, PGD**) and **Kyber-based post-quantum perturbations**.

---

## **📌 Key Features**
- ✅ **Adversarial AI Security Benchmarking** (FGSM, PGD, PQC perturbations)
- ✅ **Kyber512 Post-Quantum Encryption** to enhance AI security
- ✅ **Integrates Foolbox AI Security Library** for adversarial attacks
- ✅ **Python-based framework for reproducible benchmarking**

---

## **📂 Repository Structure**
```bash
SafeBench-PQC-Benchmark/
│── data/                           # CIFAR-10 dataset (Git LFS)
│── liboqs-python/                   # PQC library for Kyber encryption
│── countermeasures_sim.py           # Timing & fault attack simulations
│── fault_injection_sim.py           # PQC fault attack resilience tests
│── pqc_ai_robustness.py             # AI robustness benchmarking script
│── pqc_benchmark.py                 # Kyber PQC benchmarking for AI models
│── timing_attack_sim.py             # Timing attack simulation on PQC
│── .gitattributes                    # Git LFS tracking for large files
│── README.md                        # Project documentation


## 🔬 Benchmark Results

| Attack Type            | Baseline Accuracy (%) | PQC-Encrypted Accuracy (%) | Adversarial Accuracy (%) |
|------------------------|---------------------- |--------------------------- |--------------------------|
| No Attack              | 12.59                 | 10.01                      | N/A                      |
| FGSM Attack            | N/A                   | N/A                        | 0.31                     |
| PGD Attack             | N/A                   | N/A                        | 0.00                     |
| Quantum Perturbation   | N/A                   | N/A                        | 8.82                     |

Key Insight: PQC encryption enhances AI model security but introduces a slight accuracy drop.

 How to Run SafeBench-PQC-Benchmark

🔹 1️⃣ Clone the Repository  
```bash
git clone https://github.com/raviraja1218/SafeBench-PQC-Benchmark.git
cd SafeBench-PQC-Benchmark

🔹 2️⃣ Install Dependencies
pip install -r requirements.txt

🔹 3️⃣ Run AI Robustness Benchmark
python pqc_ai_robustness.py

🔹 4️⃣ Run PQC Security Benchmark
python pqc_benchmark.py



