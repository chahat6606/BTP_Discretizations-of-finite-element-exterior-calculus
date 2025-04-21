#include <iostream>
#include "eigen-3.4.0/Eigen/Dense"
#include <chrono>
Eigen::VectorXd gaussSeidelSolver(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x0,
    int max_iter,
    double tol = 1e-10)
{
    std::cout << "\n[Gauss-Seidel Method]" << std::endl;
    int n = A.rows();
    Eigen::VectorXd x = x0;

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::VectorXd x_old = x;

        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) sigma += A(i, j) * x(j); // using the updated values as we progress
            }
            x(i) = (b(i) - sigma) / A(i, i);
        }

        double diffNorm = (x - x_old).norm();
        // std::cout << "Iteration " << iter + 1 
        //           << " - 2-norm of update: " << diffNorm << std::endl;
        
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
    std::cout << "Gauss-Seidel Solver\n";
    int n;
    
    std::cout << "Enter matrix dimension n: ";
    std::cin >> n;

    double alpha;
    std::cout << "Enter diagonal value alpha: ";
    std::cin >> alpha;

    int max_iter;
    std::cout << "Enter max number of iterations: ";
    std::cin >> max_iter;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        A(i, i) = alpha;
        if (i > 0) A(i, i - 1) = -1;
        if (i < n - 1) A(i, i + 1) = -1;
    }


    Eigen::VectorXd b(n);
    for (int i = 0; i < n; ++i)
        b(i) = i;

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(n);
  
    auto start_gs = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd x_gs = gaussSeidelSolver(A, b, x0, max_iter);
    auto end_gs = std::chrono::high_resolution_clock::now();

    std::cout << "Gauss-Seidel ||Ax - b|| = " << (A * x_gs - b).norm() << std::endl;
    std::cout << "Gauss-Seidel 2-norm of x = " << x_gs.norm() << std::endl;
    std::cout << "Gauss-Seidel Time: "
              << std::chrono::duration<double, std::milli>(end_gs - start_gs).count()
              << " ms\n";

    return 0;
}
