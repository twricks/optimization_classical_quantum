import numpy as np
import matplotlib.pyplot as plt


#User input
n = int(input("Number of decision variables (n), i.e. terms in the linear function we wish to maximize: "))
m = int(input("Number of constraints (m) we wish to subject the variables to: "))

# Objective function: maximize c^t(x)
print("\nEnter objective coefficients c (space or one per line):")
c_list = []
for i in range(n):
    val = float(input(f"  c[{i+1}] = "))
    c_list.append(val)
c = np.array(c_list)

#
print("\nEnter constraints in ≤ form (A x ≤ b) (If you have a ≥ constraint, multiply the row by -1 and flip the sign of b):")
A = np.zeros((m, n))
b = np.zeros(m)

for i in range(m):
    print(f"\nConstraint {i+1}:")
    for j in range(n):
        val = float(input(f"  A[{i+1},{j+1}] = "))
        A[i, j] = val
    b[i] = float(input(f"  b[{i+1}] = "))

print("Inputs receiving, initiating algorithm..")


def simplex(A, c, b):
    m, n = A.shape
    # Tableau setup: m constraint rows + 1 objective row
    tbl = np.zeros((m + 1, n + m + 1))
    tbl[:m, :n] = A
    tbl[:m, n:n + m] = np.eye(m)          # identity matrix for slack variables
    tbl[:m, -1] = b
    tbl[-1, :n] = -c                       # negative because we are maximizing

    print("Initial tableau:\n", np.round(tbl, decimals=4))

    # Tracking for the plot
    basic = list(range(n, n + m))          # initially slacks are basic
    iterations = [0]
    obj_values = [0.0]                     
    norms = [0.0]                         

    iteration = 0
    while True:
        iteration += 1

        # Optimality check
        if np.all(tbl[-1, :-1] >= 0):
            break

        # Choose entering variable (most negative coefficient in objective row)
        pivot_col = np.argmin(tbl[-1, :-1])

        # Ratio test
        ratios = np.full(m, np.inf)
        for i in range(m):
            if tbl[i, pivot_col] > 1e-6:
                ratios[i] = tbl[i, -1] / tbl[i, pivot_col]

        if np.min(ratios) == np.inf:
            print("Problem is unbounded!")
            break

        pivot_row = np.argmin(ratios)

        # Perform the pivot operation
        pivot_element = tbl[pivot_row, pivot_col]
        tbl[pivot_row] /= pivot_element
        for i in range(m + 1):
            if i != pivot_row:
                tbl[i] -= tbl[i, pivot_col] * tbl[pivot_row]

        # Update which variable is basic in this row
        basic[pivot_row] = pivot_col

        # Extract current solution vector x (only decision variables)
        x = np.zeros(n)
        for i in range(m):
            col = basic[i]
            if col < n:                    # decision variable, not slack
                x[col] = tbl[i, -1]

        # Record for plotting
        current_norm = np.linalg.norm(x)
        current_obj = tbl[-1, -1]

        iterations.append(iteration)
        obj_values.append(current_obj)
        norms.append(current_norm)

        print(f"After iteration {iteration} tableau:\n", np.round(tbl, decimals=4))

    # Final optimal value
    optimal_value = tbl[-1, -1]
    print(f"\nOptimal objective value = {optimal_value:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value', color='tab:blue')
    ax1.plot(iterations, obj_values, 'b-o', linewidth=2.5, markersize=6, label='Objective Value')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)


    plt.title('Simplex Algorithm Progress\n')
    

    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper left')

    plt.tight_layout()
    plt.show()

    return optimal_value


optimal = simplex(A, c, b)