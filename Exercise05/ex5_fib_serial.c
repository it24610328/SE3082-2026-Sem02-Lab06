#include <stdio.h>
#include <omp.h>

int fib(int n) {
  int i, j;
  if (n<2)
    return n;
  else {
    i=fib(n-1);
    j=fib(n-2);
    return i+j;
  }
}

int main() {
  int n = 30;
  double tstart, tstop, tcalc;

  tstart = omp_get_wtime();
  int result = fib(n);
  tstop = omp_get_wtime();
  tcalc = tstop - tstart;

  printf("Fibonacci(%d) = %d\n", n, result);
  printf("Time = %f seconds\n", tcalc);

  return 0;
}
