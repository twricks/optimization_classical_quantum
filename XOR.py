import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random

# parameters
m = 2
delta = 2
iterations = 4000
correlator = 0.99 #correlation strength of x and y
#each row of counts is an output pair, each column an input pair
counts = np.zeros((4, 4), dtype=int)

sim = AerSimulator()

def Referee(correlator):
    x = random.randint(0, 1)
    if random.random() < correlator:
        y=x
    else:
        y = random.randint(0, 1)
    return x, y

def Alice(x, qc):
    if x == 0:
        qc.h(0)
    qc.measure(0, 0)

def Bob(y, qc):
    if y == 0:
        qc.ry(-np.pi/4, 1)  # (Z + X)/√2
    else:
        qc.ry(np.pi/4, 1)   # (Z - X)/√2
    qc.measure(1, 1)

for _ in range(iterations):
    qc = QuantumCircuit(2, 2)
    
    # entangled pair
    qc.h(0)
    qc.cx(0, 1)

    x, y = Referee(correlator)
    Alice(x, qc)
    Bob(y, qc)

    result = sim.run(qc,shots=200).result()
    counts_dict = result.get_counts(qc)
    bitstring = next(iter(counts_dict))

    # Qiskit uses little-endian: c1 c0 → "ba"
    b = int(bitstring[0])
    a = int(bitstring[1])

    #loading the results into counts
    input_idx = 2 * x + y
    output_idx = 2 * a + b

    counts[input_idx, output_idx] += 1

# ---- plotting ----
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(counts, cmap="coolwarm")

plt.colorbar(im, ax=ax)

ax.set_xlabel("Output pair (a,b)")
ax.set_ylabel("Input pair (x,y)")

labels = ["00", "01", "10", "11"]
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

for i in range(4):
    for j in range(4):
        ax.text(j, i, counts[i, j],
                ha="center", va="center",
                color="white" if counts[i, j] > np.max(counts)/2 else "black")

plt.tight_layout()
plt.show()