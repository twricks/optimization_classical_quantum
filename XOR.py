import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error
import random

# ========================= PARAMETERS =========================
iterations = 800
correlations = [0, 0.5, 0.66, 0.75, 0.99]
gamma = 0.0 #amplitude noise strength
lambda_phase = 0.0 #dephasing noise strength
beta1 = 0.0 #
beta2 = 0.0
gamma_vals = np.linspace(0, 0.2, 8)
lambda_vals = np.linspace(0, 0.2, 8)

heatmap = np.zeros((len(lambda_vals), len(gamma_vals)))

noise_model = NoiseModel()

# single-qubit amplitude damping
amp_error = amplitude_damping_error(gamma)
phase_error = phase_damping_error(lambda_phase)
combined_error = amp_error.compose(phase_error)
# apply to single-qubit gates
noise_model.add_all_qubit_quantum_error(combined_error, ['h', 'ry'])
noise_model.add_all_qubit_quantum_error(combined_error, ['measure'])
noise_model.add_all_qubit_quantum_error(
    combined_error.tensor(combined_error), ['cx']
)

# simulator with noise
sim = AerSimulator(noise_model=noise_model)

# ========================= REFEREE =========================
def Referee(correlator):
    x = random.randint(0, 1)
    if random.random() < correlator:
        y = x
    else:
        y = 1 - x
    return x, y

# ========================= QUANTUM STRATEGY =========================
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

# ========================= CLASSICAL STRATEGY =========================
def cAlice(x):
    return 0

def cBob(y):
    return 0

# ========================= UTILITY FUNCTION =========================
def utility(a, b, x, y, beta1, beta2):
    
    desired_parity = x ^ y
    actual_parity = a ^ b

    if desired_parity ==0:
        return 1.0 if actual_parity==0 else 0.0
    else:
        beta = beta1 if (x ==0 and y==1) else beta2
        return (1.0-beta) if actual_parity ==1 else 0.0

# ===================== STORAGE =====================
quantum_util = []
classical_util = []
advantage = []

# ===================== MAIN LOOP =====================
for corr in correlations:
    q_reward = 0.0
    c_reward = 0.0

    for _ in range(iterations):
        x, y = Referee(corr)

        # -------- QUANTUM --------
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)

        qAlice(x, qc)
        qBob(y, qc)

        result = sim.run(qc, shots=1).result()
        bitstring = next(iter(result.get_counts(qc)))

        b = int(bitstring[0])
        a = int(bitstring[1])

        q_reward += utility(a, b, x, y, beta1=beta1, beta2=beta2)

        # -------- CLASSICAL --------
        a_c = cAlice(x)
        b_c = cBob(y)

        c_reward += utility(a_c, b_c, x, y, beta1=beta1, beta2=beta2)

    q_util = q_reward / iterations
    c_util = c_reward / iterations
    adv = q_util - c_util

    quantum_util.append(q_util)
    classical_util.append(c_util)
    advantage.append(adv)

# ===================== PLOTTING =====================
plt.figure(figsize=(10, 6))

plt.plot(correlations, classical_util, 'o-', label='Classical expected utility', linewidth=2)
plt.plot(correlations, quantum_util, 's-', label='Quantum expected utility', linewidth=2)
plt.plot(correlations, advantage, '^-', label='Quantum advantage', linewidth=2, color='green')

plt.xlabel('Correlation Strength  P(x == y)')
plt.ylabel('Expected Utility')
plt.title('Generalized CHSH Game: Expected Utility vs Input Correlation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0.0, 1.0)

# value labels
for i, corr in enumerate(correlations):
    plt.text(corr, classical_util[i] + 0.01,
             f'{classical_util[i]:.3f}', ha='center', fontsize=9)
    plt.text(corr, quantum_util[i] + 0.01,
             f'{quantum_util[i]:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.show()

print("\nPlot generated successfully (expected utility version).")

fixed_corr = 0.75

for i, lambda_phase in enumerate(lambda_vals):
    for j, gamma in enumerate(gamma_vals):

        # --- noise model ---
        noise_model = NoiseModel()

        amp_error = amplitude_damping_error(gamma)
        phase_error = phase_damping_error(lambda_phase)

        combined_error = amp_error.compose(phase_error)

        noise_model.add_all_qubit_quantum_error(combined_error, ['h', 'ry'])
        noise_model.add_all_qubit_quantum_error(combined_error, ['measure'])
        noise_model.add_all_qubit_quantum_error(
            combined_error.tensor(combined_error), ['cx']
        )

        sim = AerSimulator(noise_model=noise_model)

        q_reward = 0.0
        c_reward = 0.0

        for _ in range(iterations):
            x, y = Referee(fixed_corr)

            # --- quantum ---
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)

            qAlice(x, qc)
            qBob(y, qc)

            result = sim.run(qc, shots=1).result()
            bitstring = next(iter(result.get_counts(qc)))

            b = int(bitstring[0])
            a = int(bitstring[1])

            q_reward += utility(a, b, x, y, beta1=beta1, beta2=beta2)

            # --- classical ---
            a_c = cAlice(x)
            b_c = cBob(y)

            c_reward += utility(a_c, b_c, x, y, beta1=beta1, beta2=beta2)

        q_util = q_reward / iterations
        c_util = c_reward / iterations

        heatmap[i, j] = q_util - c_util
        heatmap[i, j] = max(heatmap[i, j], 0.0)

plt.figure(figsize=(8, 6))

plt.imshow(
    heatmap,
    origin='lower',
    extent=[gamma_vals[0], gamma_vals[-1], lambda_vals[0], lambda_vals[-1]],
    aspect='auto',
    cmap='coolwarm',
    vmin=0  # important: locks zero as baseline color
)

plt.colorbar(label='Quantum Advantage')

plt.xlabel('Amplitude Damping γ')
plt.ylabel('Dephasing λ')
plt.title('Quantum Advantage Heatmap (Correlation = {fixed_corr}')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

G, L = np.meshgrid(gamma_vals, lambda_vals)

cs = plt.contour(
    G, L, heatmap,
    levels=12,
    cmap='coolwarm'
)

#plt.clabel(cs, inline=True, fontsize=8)

plt.colorbar(cs, label='Quantum Advantage')

plt.xlabel('Amplitude Damping γ')
plt.ylabel('Dephasing λ')
plt.title('Quantum Advantage Contours (Correlation = 0.5)')

plt.tight_layout()
plt.show()