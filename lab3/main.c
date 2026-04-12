#include <cblas.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    int rows;
    int cols;
    float *data;
} TMatrix;

static void abort_all(MPI_Comm comm, const char *message)
{
    fprintf(stderr, "%s\n", message);
    MPI_Abort(comm, 1);
}

static TMatrix matrix_create(int rows, int cols)
{
    TMatrix m;
    m.rows = rows;
    m.cols = cols;
    size_t count = (size_t)rows * (size_t)cols;
    m.data = count ? (float *)calloc(count, sizeof(float)) : NULL;
    return m;
}

static void matrix_free(TMatrix *m)
{
    free(m->data);
}

static void partition_1d(int total, int parts, int *counts, int *displs)
{
    int base = total / parts;
    int rem = total % parts;
    int offset = 0;

    for (int i = 0; i < parts; ++i)
    {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = offset;
        offset += counts[i];
    }
}

static int read_input_file(const char *path, TMatrix *A, TMatrix *B, int *n, int *k, int *m)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;

    if (fscanf(f, "%d %d %d", n, k, m) != 3)
    {
        fclose(f);
        return 0;
    }

    *A = matrix_create(*n, *k);
    *B = matrix_create(*k, *m);

    for (int i = 0; i < (*n) * (*k); i++)
        fscanf(f, "%f", &A->data[i]);
    for (int i = 0; i < (*k) * (*m); i++)
        fscanf(f, "%f", &B->data[i]);

    fclose(f);
    return 1;
}

static void create_column_type(int rows, int total_cols, MPI_Datatype *column_type)
{
    MPI_Datatype tmp;
    MPI_Type_vector(rows, 1, total_cols, MPI_FLOAT, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(float), column_type);
    MPI_Type_commit(column_type);
    MPI_Type_free(&tmp);
}

static TMatrix multiply_local(const TMatrix *A, const TMatrix *B)
{
    TMatrix C = matrix_create(A->rows, B->cols);

    for (int i = 0; i < A->rows; ++i)
    {
        for (int j = 0; j < B->cols; ++j)
        {
            float sum = 0.0f;
            for (int t = 0; t < A->cols; ++t)
            {
                sum += A->data[i * A->cols + t] * B->data[j * A->cols + t];
            }
            C.data[i * C.cols + j] = sum;
        }
    }

    return C;
}

static void check_with_blas(const TMatrix *A, const TMatrix *B, const TMatrix *C_parallel)
{
    int n = A->rows;
    int k = A->cols;
    int m = B->cols;

    float *blas_res = (float *)malloc(n * m * sizeof(float));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, m, k, 1.0f, A->data, k, B->data, m, 0.0f, blas_res, m);

    int match = 1;
    float eps = 0.1;
    for (int i = 0; i < n * m; i++)
    {
        if (fabsf(C_parallel->data[i] - blas_res[i]) > eps)
        {
            match = 0;
            printf("is %f, expected %f", C_parallel->data[i], blas_res[i]);
            break;
        }
    }

    if (match)
        printf("\nVerification: SUCCESS (matches BLAS)\n");
    else
        printf("\nVerification: FAILED (differs from BLAS)\n");

    free(blas_res);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 2)
    {
        if (rank == 0)
            abort_all(MPI_COMM_WORLD, "Missing input file.");
        MPI_Finalize();
        return 1;
    }

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);

    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, (int[]){0, 0}, 0, &cart);

    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);

    int row = coords[0];
    int col = coords[1];

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(cart, row, col, &row_comm);
    MPI_Comm_split(cart, col, row, &col_comm);

    TMatrix A = {0}, B = {0}, C = {0};
    int n = 0, k = 0, m = 0, ok = 1;

    if (rank == 0)
    {
        printf("rows = %d, cols = %d", dims[0], dims[1]);
        ok = read_input_file(argv[1], &A, &B, &n, &k, &m);
    }

    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok)
    {
        if (rank == 0)
            abort_all(MPI_COMM_WORLD, "Failed to read input.");
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *row_counts = malloc(dims[0] * sizeof(int));
    int *row_displs = malloc(dims[0] * sizeof(int));
    int *col_counts = malloc(dims[1] * sizeof(int));
    int *col_displs = malloc(dims[1] * sizeof(int));

    partition_1d(n, dims[0], row_counts, row_displs);
    partition_1d(m, dims[1], col_counts, col_displs);

    TMatrix A_local = matrix_create(row_counts[row], k);
    TMatrix B_local = matrix_create(k, col_counts[col]);

    if (col == 0)
    {
        int *sendcounts = malloc(dims[0] * sizeof(int));
        int *displs = malloc(dims[0] * sizeof(int));

        for (int i = 0; i < dims[0]; i++)
        {
            sendcounts[i] = row_counts[i] * k;
            displs[i] = row_displs[i] * k;
        }

        MPI_Scatterv(A.data, sendcounts, displs, MPI_FLOAT,
                     A_local.data, row_counts[row] * k, MPI_FLOAT,
                     0, col_comm);

        free(sendcounts);
        free(displs);
    }

    MPI_Bcast(A_local.data, row_counts[row] * k, MPI_FLOAT, 0, row_comm);

    MPI_Datatype col_type;
    create_column_type(k, m, &col_type);

    if (row == 0)
    {
        MPI_Scatterv(B.data, col_counts, col_displs, col_type,
                     B_local.data, k * col_counts[col], MPI_FLOAT,
                     0, row_comm);
    }

    MPI_Bcast(B_local.data, k * col_counts[col], MPI_FLOAT, 0, col_comm);
    MPI_Type_free(&col_type);

    TMatrix C_local = multiply_local(&A_local, &B_local);

    if (rank == 0)
    {
        C = matrix_create(n, m);
    }

    if (rank != 0)
    {
        MPI_Send(C_local.data, row_counts[row] * col_counts[col], MPI_FLOAT, 0, 0, cart);
    }
    else
    {
        for (int i = 0; i < row_counts[0]; i++)
        {
            for (int j = 0; j < col_counts[0]; j++)
            {
                C.data[i * C.cols + j] = C_local.data[i * C_local.cols + j];
            }
        }

        for (int r = 0; r < dims[0]; r++)
        {
            for (int c = 0; c < dims[1]; c++)
            {
                if (r == 0 && c == 0)
                    continue;

                int src_rank;
                int coords_rc[2] = {r, c};
                MPI_Cart_rank(cart, coords_rc, &src_rank);

                MPI_Datatype sub_type;
                int sizes[2] = {n, m};
                int subsizes[2] = {row_counts[r], col_counts[c]};
                int starts[2] = {row_displs[r], col_displs[c]};

                MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &sub_type);
                MPI_Type_commit(&sub_type);

                MPI_Recv(C.data, 1, sub_type, src_rank, 0, cart, MPI_STATUS_IGNORE);
                MPI_Type_free(&sub_type);
            }
        }
    }

    if (rank == 0)
    {
        check_with_blas(&A, &B, &C);

        matrix_free(&A);
        matrix_free(&B);
        matrix_free(&C);
    }

    matrix_free(&A_local);
    matrix_free(&B_local);
    matrix_free(&C_local);
    free(row_counts);
    free(row_displs);
    free(col_counts);
    free(col_displs);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart);
    MPI_Finalize();

    return 0;
}