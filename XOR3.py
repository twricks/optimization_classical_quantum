import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, phase_damping_error
import random

# ========================= PARAMETERS =========================
iterations = 4000
correlations = [0, 0.25, 0.5, 0.66, 0.75, 0.8, 1]
gamma = 0.0 #amplitude noise strength
lambda_phase = 0.0 #dephasing noise strength
beta1 = 0.0 #
beta2 = 0.0
gamma_vals = np.linspace(0, 0.20, 10)
lambda_vals = np.linspace(0, 0.2, 10)

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
    
    desired_parity = x & y
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

# ===================== FIDELITY ======================

def singlet_fidelity(sim, shots=2000):
    qc_fid = QuantumCircuit(2,2)
    qc_fid.h(0)
    qc_fid.cx(0,1)
    qc_fid.measure([0,1],[0,1])
    result = sim.run(qc_fid, shots=shots).result()
    counts = result.get_counts()
    f_singlet = (counts.get('01',0) + counts.get('10',0))/shots
    return 1.0 -f_singlet #epsilon_s in the paper
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
plt.title(
    f'Generalized CHSH Game: Expected Utility vs Input Correlation \n'
    f'$\\beta_1 = {beta1}, \\beta_2 = {beta2}, iterations = {iterations}$.'
)
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

#print("\nPlot generated successfully (expected utility version).")

fixed_corr = 0.5

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
        eps_s = singlet_fidelity(sim, shots=2000)
        eps_meas =0.0 
        eps_combined = 1 - (1-4*eps_s/3) * (1-eps_meas)**2
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

heatmap = np.clip(heatmap, 1e-3, None)

# ===================== HEATMAP PLOT =====================
plt.figure(figsize=(8, 6))

plt.imshow(
    heatmap,
    origin='lower',
    extent=[gamma_vals[0], gamma_vals[-1], lambda_vals[0], lambda_vals[-1]],
    aspect='auto',
    cmap='coolwarm',
    norm=LogNorm(vmin=1e-3, vmax=heatmap.max())
)

plt.colorbar(label='Quantum Advantage (log scale)')
plt.xlabel('Amplitude Damping γ')
plt.ylabel('Dephasing λ')
plt.title(f'Quantum Advantage Heatmap (Correlation = {fixed_corr})')

plt.tight_layout()
plt.show()

# ===================== CONTOUR =====================
plt.figure(figsize=(8, 6))

G, L = np.meshgrid(gamma_vals, lambda_vals)

cs = plt.contour(
    G, L, heatmap,
    levels=np.logspace(-3, np.log10(heatmap.max()), 12),
    norm=LogNorm(vmin=1e-3, vmax=heatmap.max()),
    cmap='coolwarm'
)

plt.colorbar(cs, label='Quantum Advantage (log scale)')
plt.xlabel('Amplitude Damping γ')
plt.ylabel('Dephasing λ')
plt.title(f'Quantum Advantage Contours (Correlation = {fixed_corr})')

plt.tight_layout()
plt.show()

# ===================== NEW FIGURE: Quantum Advantage vs. Combined Infidelity ε (Fig. 2c, β=0 only) =====================

# ===================== HELPER FUNCTION (uses global iterations) =====================
def compute_advantage_and_eps(noise_model, shots_fid=2000):
    sim = AerSimulator(noise_model=noise_model)
   
    eps_s = singlet_fidelity(sim, shots=shots_fid)
    eps_meas = 0.0
    eps_combined = 1 - (1 - 4*eps_s/3) * (1 - eps_meas)**2   # paper Eq. (30)
   
    q_reward = 0.0
    c_reward = 0.0
    for _ in range(iterations):          # ← uses your global iterations variable
        x, y = Referee(0.5)
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
        a_c = cAlice(x)
        b_c = cBob(y)
        c_reward += utility(a_c, b_c, x, y, beta1=beta1, beta2=beta2)
   
    advantage = (q_reward - c_reward) / iterations
    return eps_combined, advantage


# ===================== ANALYTIC WERNER LINE (paper Eq. 36) =====================
def paper_advantage(eps):
    return ((1 - eps) * np.sqrt(2) - 1) / 4


# ===================== SWEEPS =====================
n_points = 12
eps_paper = np.linspace(0, 0.35, 200)
adv_paper = paper_advantage(eps_paper)

# Pure dephasing only
lambda_sweep = np.linspace(0, 0.25, n_points)
eps_deph, adv_deph = [], []
for lam in lambda_sweep:
    noise_model = NoiseModel()
    phase_error = phase_damping_error(lam)
    noise_model.add_all_qubit_quantum_error(phase_error, ['h','ry','measure'])
    noise_model.add_all_qubit_quantum_error(phase_error.tensor(phase_error), ['cx'])
    eps, adv = compute_advantage_and_eps(noise_model)   # ← no extra arguments needed
    eps_deph.append(eps)
    adv_deph.append(adv)

# Pure amplitude damping only
gamma_sweep = np.linspace(0, 0.25, n_points)
eps_amp, adv_amp = [], []
for gam in gamma_sweep:
    noise_model = NoiseModel()
    amp_error = amplitude_damping_error(gam)
    noise_model.add_all_qubit_quantum_error(amp_error, ['h','ry','measure'])
    noise_model.add_all_qubit_quantum_error(amp_error.tensor(amp_error), ['cx'])
    eps, adv = compute_advantage_and_eps(noise_model)   # ← no extra arguments needed
    eps_amp.append(eps)
    adv_amp.append(adv)


# ===================== PLOT =====================
plt.figure(figsize=(9, 6))
plt.plot(eps_paper, adv_paper, 'r--', linewidth=3, label='Paper: isotropic Werner noise (analytic)')
plt.plot(eps_deph, adv_deph, 'o-', color='blue', linewidth=2, markersize=6,
         label='Simulation: pure dephasing (λ only)')
plt.plot(eps_amp, adv_amp, 's-', color='green', linewidth=2, markersize=6,
         label='Simulation: pure amplitude damping (γ only)')

plt.xlabel(r'Combined infidelity $\varepsilon$')
plt.ylabel(r'Quantum advantage $\Delta\omega$')
plt.title(r'Quantum advantage vs. $\varepsilon$ — CHSH ($\beta_1=\beta_2=0$, $P(x,y)=1/4$)')
plt.yscale('log')
plt.ylim(1e-4, 0.12)
plt.xlim(0, 0.35)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

print("✅ Fig. 2(c) reproduction complete (uses your global iterations).")
print(f"Paper isotropic threshold (Δω = 0): ε ≈ {1-1/np.sqrt(2):.4f}")
