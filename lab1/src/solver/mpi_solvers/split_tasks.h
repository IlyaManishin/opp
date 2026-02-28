#pragma once

void mat_vec_task(
    int rank,
    int size,
    int *displs,
    int maxLocal,
    int n,

    double *mat_part,
    double *v_part,
    double *d_part,
    double *d_buf);