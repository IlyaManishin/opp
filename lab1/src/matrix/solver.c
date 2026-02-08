#include "solver.h"
#include "matrix.h"

#include <cblas.h>
#include <math.h>
#include <stdlib.h>

static double vec_dot(const double *u, const double *v, int n)
{
    return cblas_ddot(n, u, 1, v, 1);
}

static double vec_norm(const double *v, int n)
{
    return sqrt(vec_dot(v, v, n));
}

static bool check_params(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters)
{
    if (!A || !b || !x)
        return false;
    if (n <= 0 || eps <= 0.0 || max_iters <= 0)
        return false;
    return true;
}

SolverStatus solve_linear_single(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters)
{
    if (check_params(A, n, b, x, eps, max_iters))
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

    for (int iter = 0; iter < max_iters; iter++)
    {
        matrix_mul_vec(A, n, n, x, y);
        for (int i = 0; i < n; i++)
        {
            y[i] -= b[i];
        }

        double norm_r = vec_norm(y, n);
        if (norm_r / norm_b < eps)
        {
            free(y);
            free(Ay);
            return SOL_OK;
        }

        matrix_mul_vec(A, n, n, y, Ay);

        double den = vec_dot(Ay, Ay, n);
        double num = vec_dot(y, Ay, n);

        if (fabs(num) < 1e-30)
        {
            free(y);
            free(Ay);
            return SOL_INVALID;
        }

        double tau = num / den;

        for (int i = 0; i < n; i++)
        {
            x[i] -= tau * y[i];
        }
    }

    free(y);
    free(Ay);
    return SOL_MAX_ITERS;
}

SolverStatus solve_linear_multy(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters)
{
    if (check_params(A, n, b, x, eps, max_iters))
        return SOL_INPUT_ERR;
    
}