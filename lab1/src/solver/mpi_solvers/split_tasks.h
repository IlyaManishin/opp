#pragma once

void mat_vec_mul_task(
    int rank,
    int size,
    int *displs,
    int maxLocal,
    int n,

    double *mat_part,
    double *v_part,
    double *d_part,
    double *d_buf);

double vec_dot_task(
    double *v1_part,
    double *v2_part,
    int localCount);