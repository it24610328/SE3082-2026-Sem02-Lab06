#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000000
#define STRIP_SIZE 8

int main() {
    double *A = malloc(N * sizeof(double));
    double *B = malloc(N * sizeof(double));
    double *C = malloc(N * sizeof(double));
    double tstart, tstop, tcalc;

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0;
        B[i] = i * 2.0;
    }

    tstart = omp_get_wtime();

    #pragma omp parallel for
    for (int strip = 0; strip < N; strip += STRIP_SIZE) {
        int end = strip + STRIP_SIZE;
        if (end > N) end = N;
        #pragma omp simd
        for (int i = strip; i < end; i++) {
            C[i] = A[i] * B[i];
        }
    }

    tstop = omp_get_wtime();
    tcalc = tstop - tstart;

    printf("Threads used: %d\n", omp_get_max_threads());
    printf("C[0] = %f, C[500000] = %f, C[999999] = %f\n", C[0], C[500000], C[999999]);
    printf("Time = %f seconds\n", tcalc);

    free(A);
    free(B);
    free(C);
    return 0;
}
