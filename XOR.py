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
#each row of counts is an output pair, each column an input pair
q_counts = np.zeros((4, 4), dtype=int)
c_counts = np.zeros((4, 4), dtype=int)

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

for corr in correlations:
    q_wins = 0
    c_wins = 0
    for _ in range(iterations):
        qc = QuantumCircuit(2, 2)
    
        # entangled pair
        qc.h(0)
        qc.cx(0, 1)

        x, y = Referee(correlator)
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



print(f"Correlation strength: {correlator}")
print(f"Quantum win rate: {quantum_win_rate}")
print(f"Classical win rate: {classical_win_rate}")
print(f"Quantum Advantage: {quantum_win_rate-classical_win_rate}")
# ---- plotting ----
#fig, ax = plt.subplots(figsize=(6, 5))
#im = ax.imshow(q_counts, cmap="coolwarm")

#plt.colorbar(im, ax=ax)

#ax.set_xlabel("Output pair (a,b)")
#ax.set_ylabel("Input pair (x,y)")

#labels = ["00", "01", "10", "11"]
#ax.set_xticks(range(4))
#ax.set_yticks(range(4))
#ax.set_xticklabels(labels)
#ax.set_yticklabels(labels)

#for i in range(4):
 #   for j in range(4):
  #      ax.text(j, i, q_counts[i, j],
   #             ha="center", va="center",
    #            color="white" if q_counts[i, j] > np.max(q_counts)/2 else "black")

#plt.tight_layout()
#plt.show()