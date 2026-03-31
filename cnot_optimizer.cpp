#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <fstream>
#include <vector>

using namespace Eigen;
using namespace std::complex_literals;

int main() {
    constexpr int N = 4;  // 2-qubit system
    constexpr int max_iters = 3000;
    double perturber = 0.05;
    double T0 = 1.0;

    // Target CNOT gate
    MatrixXcd CNOT(N, N);
    CNOT << 1,0,0,0,
            0,1,0,0,
            0,0,0,1,
            0,0,1,0;

    // Random initial operator
    MatrixXcd A = MatrixXcd::Random(N, N) + 1i * MatrixXcd::Random(N, N);

    // Defining computational basis states and their CNOT outputs
    std::vector<VectorXcd> inputs(N), targets(N);
    for (int i = 0; i < N; ++i) inputs[i] = VectorXcd::Unit(N, i);

    // CNOT action
    targets[0] = VectorXcd::Unit(N, 0);  // |00> -> |00>
    targets[1] = VectorXcd::Unit(N, 1);  // |01> -> |01>
    targets[2] = VectorXcd::Unit(N, 3);  // |10> -> |11>
    targets[3] = VectorXcd::Unit(N, 2);  // |11> -> |10>

    // Cost function defining how far the matrix is from CNOT
    auto compute_cost = [&](const MatrixXcd& M) {
        double cost = 0.0;
        for (int i = 0; i < N; ++i) {
            VectorXcd result = M * inputs[i];
            cost += (result - targets[i]).squaredNorm();
        }
        return cost;
    };

    auto compute_fidelity = [&](const MatrixXcd& M) {
        double total_fid = 0.0;
        for (int i = 0; i < N; ++i) {
            VectorXcd result = M * inputs[i];
            total_fid += std::abs(result.normalized().dot(targets[i].normalized()));
        }
        return total_fid / N;
    };

    //finding how much our randomly generated matrix fails to be unitary
    auto unitary_error = [](const MatrixXcd& A) {
        MatrixXcd I = MatrixXcd::Identity(A.rows(), A.cols());
        return (A.adjoint() * A - I).norm();
    };

    double best_cost = compute_cost(A);
    double initial_fidelity = compute_fidelity(A);
    std::vector<double> costs;
    costs.push_back(best_cost);
    std::vector<double> unitary_errors;
    unitary_errors.push_back(unitary_error(A));

    std::cout << "Initial cost: " << best_cost 
              << ", Initial average fidelity: " << initial_fidelity 
              << ", Initial unitary error: " << unitary_error(A) << "\n";

    for (int iter = 1; iter <= max_iters; ++iter) {
        perturber = ((double)rand() / RAND_MAX);
        MatrixXcd perturb = perturber * (MatrixXcd::Random(N, N) + 1i*MatrixXcd::Random(N, N));
        MatrixXcd A_new = A + perturb;
        double new_cost = compute_cost(A_new);
        double fidelity = compute_fidelity(A_new);

        // Accept if cost improves
        if (new_cost < best_cost) {
            A = A_new;
            best_cost = new_cost;
        }

        costs.push_back(best_cost);
        unitary_errors.push_back(unitary_error(A));

        // Stop for some fidelity threshold
        if (fidelity >= 0.95) {
            std::cout << "Target fidelity reached at iteration " << iter << "\n";
            break;
        }
    }

    // Saving costs and unitary errors over iterations to a .csv
    std::ofstream ofs("cost_trajectory.csv");
    for (size_t i = 0; i < costs.size(); ++i) {
        ofs << i << "," << costs[i] << "," << unitary_errors[i] << "\n";
    }
    ofs.close();

    double final_fidelity = compute_fidelity(A);
    std::cout << "Final cost: " << best_cost << "\n";
    std::cout << "Final matrix A (real only):\n" << A << "\n";
    std::cout << "Final average fidelity: " << final_fidelity << "\n";
    std::cout << "Final unitary error: " << unitary_error(A) << "\n";

    return 0;
}