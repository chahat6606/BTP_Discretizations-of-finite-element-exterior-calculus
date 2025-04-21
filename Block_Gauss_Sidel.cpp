#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <chrono>

using namespace Eigen;

// Block Gauss-Seidel Solver
VectorXd blockGaussSeidelSolver(
    const MatrixXd& A,
    const VectorXd& b,
    const VectorXd& x0,
    int block_size,
    int max_iter,
    double tol = 1e-10)
{
    std::cout << "\n[Block Gauss-Seidel Method]" << std::endl;
    int n = A.rows();
    VectorXd x = x0;

    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd x_old = x;

        for (int block_start = 0; block_start < n; block_start += block_size) {
            int block_end = std::min(block_start + block_size, n);
            MatrixXd A_block = A.block(block_start, block_start, block_end - block_start, block_end - block_start);
            VectorXd b_block = b.segment(block_start, block_end - block_start);

            VectorXd sigma = VectorXd::Zero(block_end - block_start);
            for (int i = block_start; i < block_end; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (j < block_start || j >= block_end)
                        sigma(i - block_start) += A(i, j) * x(j);
                }
            }

            x.segment(block_start, block_end - block_start) = A_block.ldlt().solve(b_block - sigma);
        }

        double diffNorm = (x - x_old).norm();
        if (diffNorm < tol) {
            std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
            return x;
        }
    }

    std::cout << "Did not converge in " << max_iter << " iterations." << std::endl;
    return x;
}

int main()
{
    std::cout << "Block Jacobi and Gauss-Seidel Solver\n";
    int n;
    std::cout << "Enter matrix dimension n: ";
    std::cin >> n;

    double alpha;
    std::cout << "Enter diagonal value alpha: ";
    std::cin >> alpha;

    int max_iter;
    std::cout << "Enter max number of iterations: ";
    std::cin >> max_iter;

    int block_size=16;

    MatrixXd A = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = alpha;
        if (i > 0) A(i, i - 1) = -1;
        if (i < n - 1) A(i, i + 1) = -1;
    }

    VectorXd b(n);
    for (int i = 0; i < n; ++i)
        b(i) = i;

    VectorXd x0 = VectorXd::Zero(n);


    auto start_bgs = std::chrono::high_resolution_clock::now();
    VectorXd x_bgs = blockGaussSeidelSolver(A, b, x0, block_size, max_iter);
    auto end_bgs = std::chrono::high_resolution_clock::now();

    std::cout << "Block Gauss-Seidel ||Ax - b|| = " << (A * x_bgs - b).norm() << std::endl;
    std::cout << "Block Gauss-Seidel 2-norm of x = " << x_bgs.norm() << std::endl;
    std::cout << "Block Gauss-Seidel Time: "
              << std::chrono::duration<double, std::milli>(end_bgs - start_bgs).count()
              << " ms\n";

    return 0;
}
