#pragma once

#define LOCAL_COUNT(displs, rank, size, n) \
    ((rank) == (size) - 1 ? (n) - (displs)[rank] : (displs)[(rank) + 1] - (displs)[rank])

#define max(a, b) ((a) < (b)? (b) : (a))

bool check_params(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters);
