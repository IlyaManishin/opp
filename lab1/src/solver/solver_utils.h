#pragma once

double vec_dot(const double *u, const double *v, int n);

double vec_norm(const double *v, int n);

bool check_params(
    const double *A,
    int n,
    const double *b,
    double *x,
    double eps,
    int max_iters);