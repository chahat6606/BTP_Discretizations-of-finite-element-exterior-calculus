#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <chrono>

using namespace Eigen;

// Block Jacobi Solver
VectorXd blockJacobiSolver(
    const MatrixXd& A,
    const VectorXd& b,
    const VectorXd& x0,
    int block_size,
    int max_iter,
    double tol = 1e-10)
{
    std::cout << "\n[Block Jacobi Method]" << std::endl;
    int n = A.rows();
    VectorXd x = x0;
    VectorXd x_new = x0;

    for (int iter = 0; iter < max_iter; ++iter) {
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

            x_new.segment(block_start, block_end - block_start) = A_block.ldlt().solve(b_block - sigma);
        }

        double diffNorm = (x_new - x).norm();
        if (diffNorm < tol) {
            std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
            return x_new;
        }

        x = x_new;
    }

    std::cout << "Did not converge in " << max_iter << " iterations." << std::endl;
    return x;
}


int main()
{
    std::cout << "Block Jacobi Solver\n";
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

    auto start_bj = std::chrono::high_resolution_clock::now();
    VectorXd x_bj = blockJacobiSolver(A, b, x0, block_size, max_iter);
    auto end_bj = std::chrono::high_resolution_clock::now();

    std::cout << "Block Jacobi ||Ax - b|| = " << (A * x_bj - b).norm() << std::endl;
    std::cout << "Block Jacobi 2-norm of x = " << x_bj.norm() << std::endl;
    std::cout << "Block Jacobi Time: "
              << std::chrono::duration<double, std::milli>(end_bj - start_bj).count()
              << " ms\n";


    return 0;
}
