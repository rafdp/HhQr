#include <iostream>
#include "Eigen/Dense"
#include <string>
#include <stdio.h>
using Eigen::MatrixXd;
using Eigen::VectorXd;

const double Tolerance = 1e-12;

int sgn(double val) {
    return (Tolerance < val) - (val < Tolerance);
}
double Clean_zeroes(double x)
{
    if (fabs(x) < Tolerance)
        return 0.0;
    return x;
}

bool QRDecomposition(const MatrixXd& A, MatrixXd* Q, MatrixXd* R)
{ 
    const int cols = A.cols();
    const int dim = A.rows();
    if (cols > dim) return false;
    MatrixXd H_accum = MatrixXd::Identity(dim, dim);
    MatrixXd H_curr(dim, dim);
    MatrixXd A_cpy (A);
    // Full column rank implies dim >= cols
    for (int i = 0; i < cols; i++)
    {
        H_curr = MatrixXd::Identity(dim, dim);
        VectorXd v1 (A_cpy.block(i, i, dim - i, 1));
        v1(0) += sgn(v1(0))*v1.norm();
        H_curr.block(i, i, dim - i, dim - i) -= 2*(v1*v1.transpose()) / (v1.transpose()*v1);
        H_accum *= H_curr;
        A_cpy = H_curr * A_cpy;
    }

    A_cpy = A_cpy.unaryExpr(&Clean_zeroes);
    *Q = H_accum.leftCols(cols);
    *R = A_cpy.topRows(cols);
    return true;
    
}
// Rx = b
bool SolveTriangular(const MatrixXd& R, const MatrixXd& b, VectorXd* x)
{
    const int N = R.cols();
    if (N != R.rows() || b.rows() != N) return false;
    VectorXd res(N);
    for (int i = N - 1; i >= 0; i--)
    {
        res(i) = b(i);
        for (int j = i + 1; j < N; j++)
        {
            res(i) -= res(j)*R(i, j);
        }
        if (fabs(R(i, i)) < Tolerance) return false; 
        res(i) /= R(i, i);
    }
    *x = res;
    return true;
}

int main()
{
    MatrixXd A(5,4);
    A << 1, 7, 2, 5,
         -2, 6, 3, 5,
         1, 1, 0, 5,
         9, 6, 4, 7,
         11, 52, 3, 0;
    VectorXd b(5);
    b << 1, 7, 19, 4, -12;
    printf("Matrix: \n");
    std::cout << A << std::endl;
    printf("Vector: \n");
    std::cout << b << std::endl;
    VectorXd x;
    MatrixXd Q, R;
    bool res = QRDecomposition(A, &Q, &R);
    if (!res)
    {
        printf("Could not decompose matrix\n");
    }
    //printf("Q:\n");
    //std::cout << Q << std::endl;
    //printf("R:\n");
    //std::cout << R << std::endl;
    printf("Relative decomposition error: %g %%\n", 100.0*(A - Q*R).norm() / A.norm());
    res = SolveTriangular(R, Q.transpose()*b, &x);
    if (!res)
    {
        printf("Could not solve Rx = Q^Tb");
    }
    printf("x: \n");
    std::cout << x << std::endl;
    printf("Eigen solution:\n");
    VectorXd stdsolution = A.householderQr().solve(b);
    std::cout << stdsolution << std::endl;
    printf("Custom: |Ax-b|: %g\n", (A*x-b).norm());    
    printf("Eigen: |Ax-b|: %g\n", (A*stdsolution-b).norm());    
}
