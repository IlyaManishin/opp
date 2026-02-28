#include "solver.h"
#include "utils.h"

#include "io_utils.h"
#include "matrix.h"

#include <math.h>
#include <stdlib.h>

SolverStatus solve_linear_single_impl(
    TLinearSystem linSys,
    double *x,
    double eps,
    int maxIters)
{
    double *A = linSys.A;
    double *b = linSys.b;
    int n = linSys.n;

    if (!check_params(A, n, b, x, eps, maxIters))
        return SOL_INPUT_ERR;

    double *y = vector_create(n);
    double *Ay = vector_create(n);

    if (!y || !Ay)
    {
        free(y);
        free(Ay);
        return SOL_INVALID;
    }

    double norm_b = vec_norm(b, n);
    if (norm_b < 1e-30)
        norm_b = 1.0;

    for (int iter = 0; iter < maxIters; iter++)
    {
        matrix_mul_vec(A, n, n, n, x, y);
        for (int i = 0; i < n; i++)
            y[i] -= b[i];

        double norm_y = vec_norm(y, n);
        if (norm_y / norm_b < eps)
        {
            vector_free(y);
            vector_free(Ay);
            return SOL_OK;
        }

        matrix_mul_vec(A, n, n, n, y, Ay);

        double den = vec_dot(Ay, Ay, n);
        double num = vec_dot(y, Ay, n);

        if (fabs(num) < 1e-30)
        {
            vector_free(y);
            vector_free(Ay);
            return SOL_INVALID;
        }

        double tau = num / den;

        for (int i = 0; i < n; i++)
            x[i] -= tau * y[i];
    }

    vector_free(y);
    vector_free(Ay);
    return SOL_MAX_ITERS;
}
