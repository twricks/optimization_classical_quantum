import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random

# parameters
m = 2
delta = 2
iterations = 4000
correlations = [0.25,0.5,0.75,0.8,0.99] #correlation strength of x and y; 0.5 -> independent, <0.5 means negative correlation, >0.5 positive correlation. 


sim = AerSimulator()

def Referee(correlator):
    x = random.randint(0, 1)
    if random.random() < correlator:
        y=x
    else:
        y = 1 - x
    return x, y

def qAlice(x, qc):
    if x == 0:
        pass
    else:
        qc.h(0)
    qc.measure(0, 0) #we don't need a return statement because Alice has saved her result in the first classical bit

def qBob(y, qc):
    if y == 0:
        qc.ry(-np.pi/4, 1)  # (Z + X)/√2
    else:
        qc.ry(np.pi/4, 1)   # (Z - X)/√2
    qc.measure(1, 1)

def cAlice(x):
    return x

def cBob(y):
    #classical strategy- they buy nothing. This succeeds 75% of the time, since addmition mod 2 of the outputs has to match AND condition of the inputs
    return y

#utility function block
def is_win(a,b,x,y):
    return 1 if (a ^ b) == (x & y) else 0

quantum_rates = []
classical_rates = []
advantages = []

for corr in correlations:
    q_wins = 0
    c_wins = 0
    #each row of counts is an output pair, each column an input pair
    q_counts = np.zeros((4, 4), dtype=int)
    c_counts = np.zeros((4, 4), dtype=int)

    for _ in range(iterations):
        qc = QuantumCircuit(2, 2)
    
        # entangled pair
        qc.h(0)
        qc.cx(0, 1)

        x, y = Referee(corr)
        qAlice(x, qc)
        qBob(y, qc)

        result = sim.run(qc,shots=1).result()
        counts_dict = result.get_counts(qc)
        bitstring = next(iter(counts_dict))

        # Qiskit uses little-endian: c1 c0 → "ba"
        b = int(bitstring[0])
        a = int(bitstring[1])

        #loading the results into counts
        input_idx = 2 * x + y
        output_idx = 2 * a + b

        q_counts[input_idx, output_idx] += 1
        q_wins += is_win(a,b,x,y)

        #now classical round for comparison
        a_c = cAlice(x)
        b_c = cBob(y)
        output_idx_c = 2 * a_c * b_c
        c_counts[input_idx,output_idx_c] += 1
        c_wins += is_win(a_c,b_c,x,y)

quantum_win_rate = q_wins / iterations
classical_win_rate = c_wins / iterations
adv = quantum_win_rate - classical_win_rate

quantum_rates.append(quantum_win_rate)
classical_rates.append(classical_win_rate)
advantages.append(adv)

plt.figure(figsize=(10,6))

plt.plot(correlations, classical_rates, 'o-', label='Classical',linewidth=2)
plt.plot(correlations, quantum_rates, 's-', label='Quantum(CHSH)', linewidth=2)
plt.plot(correlations, advantages, '^-', label = 'Quantum Advantage', linewidth=2, color='coolwarm')

plt.xlabel('Correlation Strength P(x==y)')
plt.ylabel('Win rate')
plt.title('Win rates vs. Input correlation')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0.4, 0.95)

for i, corr in enumerate(correlations):
    plt.text(corr, classical_rates[i] +0.01, f'{classical_rates[i]}',ha='center')
    plt.text(corr, quantum_rates[i] + 0.01, f'{quantum_rates[i]}', ha='center')

plt.tight_layout()
plt.show()


print(f"Correlation strength: {correlator}")
print(f"Quantum win rate: {quantum_win_rate}")
print(f"Classical win rate: {classical_win_rate}")
print(f"Quantum Advantage: {quantum_win_rate-classical_win_rate}")
