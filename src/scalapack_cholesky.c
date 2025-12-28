#include <mpi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void Cblacs_pinfo(int* mypnum, int* nprocs);
extern void Cblacs_get(int context, int request, int* value);
extern void Cblacs_gridinit(int* context, const char* order, int nprow, int npcol);
extern void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
extern void Cblacs_gridexit(int context);
extern void Cblacs_exit(int error_code);

extern int numroc_(const int* n, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);
extern void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
                      const int* irsrc, const int* icsrc, const int* ictxt,
                      const int* lld, int* info);
extern void pdpotrf_(const char* uplo, const int* n, double* a, const int* ia, const int* ja,
                     const int* desca, int* info);

static void parse_args(int argc, char** argv, int* n, int* nb, int* p, int* q, int* iters) {
    *n = 1024;
    *nb = 256;
    *p = 1;
    *q = 1;
    *iters = 3;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            *n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nb") == 0 && i + 1 < argc) {
            *nb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--p") == 0 && i + 1 < argc) {
            *p = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q") == 0 && i + 1 < argc) {
            *q = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            *iters = atoi(argv[++i]);
        }
    }
}

static int local_to_global(int local_index, int nb, int proc_coord, int nprocs) {
    int block = local_index / nb;
    int offset = local_index % nb;
    return block * nb * nprocs + proc_coord * nb + offset;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int n = 0, nb = 0, p = 0, q = 0, iters = 0;
    parse_args(argc, argv, &n, &nb, &p, &q, &iters);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (p * q != size) {
        if (rank == 0) {
            fprintf(stderr, "Process grid %dx%d does not match MPI size %d\n", p, q, size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int context = 0;
    Cblacs_get(0, 0, &context);
    Cblacs_gridinit(&context, "Row", p, q);

    int nprow = 0, npcol = 0, myrow = 0, mycol = 0;
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    int rsrc = 0, csrc = 0;
    int local_rows = numroc_(&n, &nb, &myrow, &rsrc, &nprow);
    int local_cols = numroc_(&n, &nb, &mycol, &csrc, &npcol);
    int lld = local_rows > 0 ? local_rows : 1;

    int descA[9];
    int info = 0;
    descinit_(descA, &n, &n, &nb, &nb, &rsrc, &csrc, &context, &lld, &info);
    if (info != 0) {
        if (rank == 0) {
            fprintf(stderr, "descinit failed with info=%d\n", info);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t local_elems = (size_t)local_rows * (size_t)local_cols;
    double* A = (double*)malloc(local_elems * sizeof(double));
    double* Aorig = (double*)malloc(local_elems * sizeof(double));
    if (!A || !Aorig) {
        if (rank == 0) {
            fprintf(stderr, "Allocation failed\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int j = 0; j < local_cols; ++j) {
        int global_j = local_to_global(j, nb, mycol, npcol);
        for (int i = 0; i < local_rows; ++i) {
            int global_i = local_to_global(i, nb, myrow, nprow);
            double val = (global_i == global_j) ? (double)n : 1e-3;
            Aorig[j * local_rows + i] = val;
        }
    }

    double total_time = 0.0;
    for (int iter = 0; iter < iters; ++iter) {
        memcpy(A, Aorig, local_elems * sizeof(double));
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        int ia = 1, ja = 1;
        pdpotrf_("L", &n, A, &ia, &ja, descA, &info);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        if (info != 0) {
            if (rank == 0) {
                fprintf(stderr, "pdpotrf failed with info=%d\n", info);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        total_time += (t1 - t0);
    }

    double avg_time = total_time / (double)iters;
    double max_time = 0.0;
    MPI_Reduce(&avg_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double time_ms = max_time * 1000.0;
        printf("{\"method\":\"scalapack\",\"n\":%d,\"iters\":%d,\"time_ms\":%.6f}\n",
               n, iters, time_ms);
    }

    free(A);
    free(Aorig);
    Cblacs_gridexit(context);
    Cblacs_exit(0);
    MPI_Finalize();
    return 0;
}
