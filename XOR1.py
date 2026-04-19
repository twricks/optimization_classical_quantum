import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random

# ========================= PARAMETERS =========================
iterations = 8000
correlations = [0.25, 0.5, 0.75, 0.8, 0.99]

sim = AerSimulator()

def Referee(correlator):
    x = random.randint(0, 1)
    if random.random() < correlator:
        y = x
    else:
        y = 1 - x
    return x, y

def qAlice(x, qc):
    if x == 1:
        qc.h(0)
    qc.measure(0, 0)

def qBob(y, qc):
    if y == 0:
        qc.ry(-np.pi/4, 1)
    else:
        qc.ry(np.pi/4, 1)
    qc.measure(1, 1)

def cAlice(x):
    return x

def cBob(y):
    return y

def is_win(a, b, x, y):
    return 1 if (a ^ b) == (x & y) else 0

# ===================== STORAGE FOR PLOTTING =====================
quantum_rates = []
classical_rates = []
advantages = []

print("\n=== CHSH Win Rates vs Correlation ===")
print(f"{'Correlation':>12} | {'Classical':>10} | {'Quantum':>10} | {'Advantage':>10}")
print("-" * 55)

for corr in correlations:
    q_wins = 0
    c_wins = 0

    for _ in range(iterations):
        x, y = Referee(corr)
        input_idx = 2 * x + y

        # --- Quantum ---
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qAlice(x, qc)
        qBob(y, qc)
        result = sim.run(qc, shots=1).result()
        bitstring = next(iter(result.get_counts(qc)))
        b = int(bitstring[0])
        a = int(bitstring[1])

        output_idx = 2 * a + b
        q_wins += is_win(a, b, x, y)

        # --- Classical ---
        a_c = cAlice(x)
        b_c = cBob(y)
        output_idx_c = 2 * a_c + b_c          # ← Fixed: + not *
        c_wins += is_win(a_c, b_c, x, y)

    q_rate = q_wins / iterations
    c_rate = c_wins / iterations
    adv = q_rate - c_rate

    quantum_rates.append(q_rate)
    classical_rates.append(c_rate)
    advantages.append(adv)

    print(f"{corr:12.2f} | {c_rate:10.4f} | {q_rate:10.4f} | {adv:10.4f}")

# ===================== PLOTTING =====================
plt.figure(figsize=(10, 6))

plt.plot(correlations, classical_rates, 'o-', label='Classical (a=x, b=y)', linewidth=2)
plt.plot(correlations, quantum_rates, 's-', label='Quantum (CHSH)', linewidth=2)
plt.plot(correlations, advantages, '^-', label='Quantum Advantage', linewidth=2, color='green')

plt.xlabel('Correlation Strength  P(x == y)')
plt.ylabel('Win Rate')
plt.title('CHSH Game: Win Rate vs Input Correlation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0.4, 0.95)

# Add value labels on the points
for i, corr in enumerate(correlations):
    plt.text(corr, classical_rates[i] + 0.008, f'{classical_rates[i]:.3f}', ha='center', fontsize=9)
    plt.text(corr, quantum_rates[i] + 0.008, f'{quantum_rates[i]:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\nPlot generated successfully!")
