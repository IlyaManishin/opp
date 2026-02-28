#include "matrix.h"

#include <math.h>

bool check_params(
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