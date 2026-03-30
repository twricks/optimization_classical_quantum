#include <iostream>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std::complex_literals;

int main() {
    constexpr int N = 3;  // dimensionality

    // Random 3x3 complex matrix
    MatrixXcd A = MatrixXcd::Random(N, N) + 1i*MatrixXcd::Random(N, N);
    //Finding the eigenvectors of the random matrix; because it is complex we have to use the Complex Schur method
    ComplexSchur<MatrixXcd> cs;
    cs.compute(A);
    VectorXcd v0 = cs.matrixU().col(0); //extracting the first Schur vector for comparison to solution
    // defining the end state
    VectorXcd b(N);   // 
    b << 1.0+0.0i, 2.0+1.0i, 3.0+0.0i;

    // Solve A*x = b
    VectorXcd x = A.fullPivLu().solve(b); //solving the linear system
    // Print solution
    std::cout << "Matrix A:\n" << A << "\n\n";
    std::cout << "Target vector b:\n" << b << "\n\n";
    std::cout << "Computed solution x:\n" << x << "\n\n";
    std::cout << "First Schur Vector x:\n" << v0 << "\n\n";
}
    

